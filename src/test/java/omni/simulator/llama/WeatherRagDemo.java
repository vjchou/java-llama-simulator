package omni.simulator.llama;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;

/**
 * 用一隻程式範例，了解 RAG (Retrieval-Augmented Generation) 的基本流程：<br>
 * 天氣查詢 RAG 範例
 */
public class WeatherRagDemo {

    // GGUF 模型路徑
    private static final String MODEL_PATH = "D:/llm/models/gguf/translategemma-4b-it-GGUF/translategemma-4b-it-Q4_K_M.gguf";

    // 台灣城市列表
    private static final String[] TAIWAN_CITIES = { 
            "台北", "新北", "桃園", "台中", 
            "台南", "高雄", "基隆", "新竹", 
            "嘉義", "屏東", "宜蘭", "花蓮", 
            "台東", "南投", "彰化" };

    // 天氣狀況
    private static final String[] WEATHER_CONDITIONS = { "晴天", "多雲", "陰天", "小雨", "大雨", "雷陣雨" };

    // 儲存天氣資料與其向量
    private final List<WeatherData> weatherDatabase = new ArrayList<>();
    private final LlamaModel model;

    public static void main(String... args) {
        new WeatherRagDemo().run();
    }

    public WeatherRagDemo() {
        System.out.println("=== 載入模型中，請稍候... ===");
        // 模型參數設定
        ModelParameters modelParams = new ModelParameters()
                .setModel(MODEL_PATH)
                .setGpuLayers(43)
                .enableEmbedding(); // 啟用 embedding  功能
        //
        this.model = new LlamaModel(modelParams);
        System.out.println("=== 模型載入完成 ===\n");
    }

    public void run() {
        try {
            // 1. 初始化：隨機產生天氣資料並建立向量索引
            initializeWeatherData();

            // 2. 進入對話迴圈
            startChatLoop();

        } finally {
            model.close();
        }
    }

    /**
     * 隨機產生 10 筆台灣城市天氣，並計算每筆的 embedding
     */
    private void initializeWeatherData() {
        System.out.println("=== 產生天氣資料並建立向量索引 ===\n");

        Random random = new Random();
        Set<String> usedCities = new HashSet<>();

        // 隨機選 10 個不重複城市
        while (usedCities.size() < 10) {
            usedCities.add(TAIWAN_CITIES[random.nextInt(TAIWAN_CITIES.length)]);
        }

        for (String city : usedCities) {
            int temperature = random.nextInt(20) + 20; // 20~39°C
            int humidity = random.nextInt(50) + 40; // 40~89%
            String condition = WEATHER_CONDITIONS[random.nextInt(WEATHER_CONDITIONS.length)];

            // 建立描述文本
            String pattern = "%s目前天氣：氣溫 %d°C，濕度 %d%%，天氣狀況為%s。";
            String description = String.format(pattern, city, temperature, humidity, condition);

            // 計算 embedding 向量 <-- RAG 關鍵步驟
            float[] embedding = model.embed(description);

            WeatherData data = new WeatherData(city, temperature, humidity, condition, description, embedding);
            weatherDatabase.add(data);

            System.out.printf("  [%s] %d°C, %s%n", city, temperature, condition);
        }

        System.out.println("\n=== 向量索引建立完成（共 " + weatherDatabase.size() + " 筆）===\n");
    }

    /**
     * 主對話迴圈
     */
    private void startChatLoop() {
        // 輸入讀取器 (建立 UTF-8 編碼的標準輸入讀取器，支援中文輸入。)
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));

        String systemPrompt = """
                你是一個台灣天氣查詢助手。根據提供的天氣資料回答用戶問題。
                請用繁體中文回答，回答要簡潔明瞭。
                如果資料中沒有相關資訊，請誠實告知。
                """;

        System.out.println("=== 天氣查詢助手 ===");
        System.out.println("提示：可以詢問「最熱的三個城市」、「哪裡在下雨」等問題");
        System.out.println("輸入 'exit' 離開\n");

        while (true) {
            try {
                System.out.print("User: ");
                String userInput = reader.readLine();

                if (userInput == null || "exit".equalsIgnoreCase(userInput.trim())) {
                    System.out.println("再見！");
                    break;
                }
                if (userInput.trim().isEmpty()) {
                    continue;
                }

                // RAG 流程
                String response = ragQuery(userInput, systemPrompt);
                System.out.println("Assistant: " + response + "\n");

            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    /**
     * RAG 查詢流程
     * 
     * @param userQuery
     * @param systemPrompt
     * @return String
     */
    private String ragQuery(String userQuery, String systemPrompt) {
        // Step 1: 計算 [查詢敘述] 的 embedding
        float[] queryEmbedding = model.embed(userQuery);

        // Step 2: 計算與所有天氣資料的相似度，取 Top-K
        List<WeatherData> relevantData = retrieveTopK(queryEmbedding, 5);

        // Step 3: 組合 context
        String context = buildContext(relevantData);

        // Step 4: 組合完整 prompt（Gemma 3 格式）
        String fullPrompt = buildGemmaPrompt(systemPrompt, context, userQuery);

        // Step 5: 生成回答
        return generateResponse(fullPrompt);
    }

    /**
     * 向量檢索：計算餘弦相似度，返回最相關的 K 筆資料
     */
    private List<WeatherData> retrieveTopK(float[] queryEmbedding, int k) {
        return weatherDatabase.stream()
                .map(data -> new AbstractMap.SimpleEntry<>(data, cosineSimilarity(queryEmbedding, data.embedding)))
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue())) // 降序
                .limit(k)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * 餘弦相似度計算
     */
    private double cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("向量長度不一致");
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator == 0 ? 0 : dotProduct / denominator;
    }

    /**
     * 建立檢索到的資料 context
     */
    private String buildContext(List<WeatherData> dataList) {
        StringBuilder sb = new StringBuilder();
        sb.append("【天氣資料庫】\n");

        // 依溫度排序顯示，方便模型理解
        dataList.stream().sorted((a, b) -> Integer.compare(b.temperature, a.temperature))
                .forEach(data -> sb.append("- ").append(data.description).append("\n"));

        return sb.toString();
    }

    /**
     * 組合 Gemma 3 格式的 prompt (systemSpec + rag context + user query)
     * 
     * @param systemPrompt
     * @param context
     * @param userQuery
     * @return String
     */
    private String buildGemmaPrompt(String systemPrompt, String context, String userQuery) {
        return String.format("""
                <start_of_turn>user
                %s

                %s

                問題：%s
                <end_of_turn>
                <start_of_turn>model
                """, systemPrompt.trim(), context.trim(), userQuery.trim());
    }

    /**
     * 使用 LLM 生成回答
     * 
     * @param prompt
     * @return String
     */
    private String generateResponse(String prompt) {
        // 推論
        InferenceParameters inferParams = new InferenceParameters(prompt)
                .setTemperature(0.3f) // 較低溫度，回答更精確
                .setTopP(0.9f) //
                .setNPredict(256) // 最大生成 token
                .setStopStrings("<end_of_turn>", "User:", "<start_of_turn>");
        // 生成回答
        StringBuilder response = new StringBuilder();

        for (LlamaOutput output : model.generate(inferParams)) {
            response.append(output);
        }
        return response.toString().trim();
    }

    /**
     * 天氣資料類別
     */
    static class WeatherData {
        final String city; // 城市名稱
        final int temperature; // 氣溫
        final int humidity; // 濕度
        final String condition; // 天氣狀況
        final String description; // 描述文本
        final float[] embedding; // 向量表示

        WeatherData(String city, int temp, int humidity, String condition, String desc, float[] embedding) {
            this.city = city;
            this.temperature = temp;
            this.humidity = humidity;
            this.condition = condition;
            this.description = desc;
            this.embedding = embedding;
        }
    }

}