package omni.simulator.llama;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import de.kherud.llama.ModelParameters;
import de.kherud.llama.args.MiroStat;

/**
 * 交互式命令列聊天機器人
 */
public class LlamaLauncher {

    // GGUF 模型路徑
    private static final String modelPath = "D:/llm/models/gguf/translategemma-4b-it-GGUF/translategemma-4b-it-Q4_K_M.gguf";

    /**
     * 
     * @param args
     * @throws IOException
     */
    // @formatter:off
    public static void main(String... args) throws IOException {
        // 模型參數
        ModelParameters modelParams = new ModelParameters() //
                .setModel(modelPath) // 指定 GGUF 模型檔案的路徑
                .setGpuLayers(43); // 載入到 GPU 的層數

        // 系統提示詞
        String systemSpec = "This is a conversation between User and Llama, a friendly chatbot.\n"
                + "Llama is helpful, kind, honest, good at writing, and never fails to answer any "
                + "requests immediately and with precision.\n";

        // 輸入讀取器 (建立 UTF-8 編碼的標準輸入讀取器，支援中文輸入。)
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
       
        // 模型載入 (Try-with-resources) : 載入時會透過 JNI 呼叫 llama.cpp 的 llama_load_model_from_file()
        try (LlamaModel model = new LlamaModel(modelParams)) {
            System.out.print(systemSpec);
            String prompt = systemSpec;
            // 主對話迴圈
            while (true) {
                prompt += "\nUser: ";
                System.out.print("\nUser: ");
                String input = reader.readLine();  // 等待用戶輸入
                prompt += input;
                System.out.print("Llama: ");
                prompt += "\nLlama: ";
                
                // 推論參數 - prompt 傳入構造函數
                InferenceParameters inferParams = new InferenceParameters(prompt)
                        .setTemperature(0.7f)
                        .setPenalizeNl(true)       // 換行策略 (降低模型輸出換行符的機率，讓回答更緊湊)
                        .setMiroStat(MiroStat.V2)  // 採樣策略 (動態調整採樣，自動平衡生成品質與多樣性)
                        .setStopStrings("User:");  // 停止詞 (當模型生成 "User:" 時停止，避免模型自問自答)
                
                // 串流輸出（核心推理）
                // generate(): 串流生成
                // complete(): 一次性生成完整回應
                System.out.println("=== LlamaModel 回應 ===");          
                for (LlamaOutput output : model.generate(inferParams)) {
                    System.out.print(output); // 逐 token 輸出
                    prompt += output; // 累積到 prompt
                }
                //
                // 同步完整生成 - 等待全部生成完才返回
                // String response = model.complete(inferParams);
                // 文本向量嵌入 - 用於 RAG、相似度搜尋
                // float[] embedding = model.embed("Embed this"); // 向量嵌入
            }
        }
    }
    // @formatter:on

}