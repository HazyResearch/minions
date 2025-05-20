// main.js
import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// ——— CONFIGURE YOUR CLOUD LLM ENDPOINT ———
// (Replace with your own key and endpoint)
// We'll now get the API key from the input field instead of hardcoding it
const OPENAI_URL = "https://api.openai.com/v1/chat/completions";

async function callCloudSupervisor(messages, apiKey) {
  if (!apiKey) {
    throw new Error("OpenAI API key is required for cloud supervision");
  }

  const resp = await fetch(OPENAI_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages,
      // ask for a JSON output
      temperature: 0,
      response_format: { type: "json_object" },
    }),
  });

  if (!resp.ok) {
    const errorData = await resp.json();
    throw new Error(
      `OpenAI API error: ${errorData.error?.message || resp.statusText}`
    );
  }

  const j = await resp.json();
  // assume the content is JSON text
  return j.choices[0].message.content;
}

async function main() {
  const logEl = document.getElementById("log");
  const taskEl = document.getElementById("task");
  const ctxEl = document.getElementById("context");
  const apiKeyEl = document.getElementById("api_key");
  const startBtn = document.getElementById("start");

  startBtn.onclick = async () => {
    logEl.textContent = "";
    const task = taskEl.value.trim();
    const context = ctxEl.value.trim();
    const apiKey = apiKeyEl.value.trim();

    if (!apiKey) {
      logEl.textContent =
        "⚠️ Error: OpenAI API key is required for cloud supervision";
      return;
    }

    try {
      // ——— Initialize local WebLLM engine ———
      const webllm = await import("https://esm.run/@mlc-ai/web-llm");

      logEl.textContent = "Loading local WebLLM model...\n";

      const initProgressCallback = (progress) => {
        logEl.textContent = `Loading local WebLLM model: ${progress.progress}%\n`;
        console.log(`Model loading progress: ${progress.progress}%`);
      };

      const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
      const localEngine = await webllm.CreateMLCEngine(selectedModel, {
        initProgressCallback: initProgressCallback,
      });

      logEl.textContent += "Model loaded successfully!\n\n";

      // ——— Start supervisor messages ———
      let supervisorMessages = [
        {
          role: "system",
          content: `You are the Supervisor (big language model). Your task is to answer the following question using documents you cannot see directly. 
A Worker (small language model) can access those documents and will answer simple, single-step questions.

Ask the Worker only one small, specific question at a time. Use multiple steps if needed (max 5 steps), then integrate the responses to answer the original question.

Format for your question:
<think briefly about the information needed to answer the question>
Respond in JSON with {"decision":..., "message":...}`,
        },
        { role: "user", content: `Task: ${task}` },
      ];

      let finalAnswer = null;

      for (let round = 0; round < 5; round++) {
        // 1) Ask cloud supervisor
        logEl.textContent += `\n[Supervisor] Round ${round + 1}…\n`;
        try {
          const supOut = await callCloudSupervisor(supervisorMessages, apiKey);
          logEl.textContent += supOut + "\n";

          let supJSON;
          try {
            supJSON = JSON.parse(supOut);
          } catch {
            logEl.textContent +=
              "\n⚠️ Error: Failed to parse JSON from supervisor\n";
            console.error("Failed to parse JSON from supervisor");
            break;
          }

          if (supJSON.decision === "provide_final_answer") {
            finalAnswer = supJSON.answer;
            break;
          }

          // 2) Send the sub-task request to local model
          logEl.textContent += `\n[Local 🤖 Worker] Prompt: ${supJSON.message}\n`;
          const worker = await localEngine.chat.completions.create({
            messages: [
              {
                role: "system",
                content: `You are the Worker (a small model). You have access to the following context:

Read the context below and prepare to answer questions from an expert user. 
### Context
${context}

### Question
${task}`,
              },
              { role: "user", content: supJSON.message },
            ],
          });
          const workerOut = worker.choices[0].message.content;
          logEl.textContent += workerOut + "\n";

          // 3) Append worker output back to supervisor context
          supervisorMessages.push({ role: "assistant", content: workerOut });

          // Use the appropriate prompt based on whether this is the final round
          if (round === 4) {
            // Last round (0-indexed, so 4 is the 5th round)
            supervisorMessages.push({
              role: "user",
              content: `The Worker replied with:

${workerOut}

This is your final round. You must provide a final answer in JSON. No further questions are allowed.

Please respond in the following format:
<briefly think about the information you have and the question you need to answer>
\`\`\`json
{
    "decision": "provide_final_answer",
    "answer": "<your final answer>"
}
\`\`\``,
            });
          } else {
            supervisorMessages.push({
              role: "user",
              content: `Here is the response from the small language model:

### Response
${workerOut}

### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question.

Think about:
1. What information we have gathered
2. Whether it is sufficient to answer the question
3. If not sufficient, what specific information is missing
4. If sufficient, how we would calculate or derive the answer

If you have enough information or if the task is complete, write a final answer that fulfills the task:

\`\`\`json
{
    "decision": "provide_final_answer", 
    "answer": "<your answer>"
}
\`\`\`

Otherwise, if the task is not complete, request the small language model to do additional work:

\`\`\`json
{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>"
}
\`\`\``,
            });
          }
        } catch (error) {
          logEl.textContent += `\n⚠️ Error: ${error.message}\n`;
          console.error("Error in callCloudSupervisor", error);
          break;
        }
      }

      if (finalAnswer) {
        logEl.textContent += `\n🎉 Final Answer:\n${finalAnswer}`;
      } else {
        logEl.textContent += `\n⚠️ No final answer after max rounds.`;
      }
    } catch (error) {
      logEl.textContent += `\n⚠️ Error: ${error.message}\n`;
      console.error("Error in main", error);
    }
  };
}

main();
