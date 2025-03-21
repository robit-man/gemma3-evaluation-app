#!/usr/bin/env python
import os
import sys
import subprocess

# --- Self-Bootstrap Venv Setup ---
if sys.prefix == sys.base_prefix:
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    pip_executable = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip.exe")
    print("Installing dependencies...")
    subprocess.check_call([pip_executable, "install", "flask", "ollama", "requests", "selenium", "chromedriver-autoinstaller"])
    python_executable = os.path.join(venv_dir, "bin", "python") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "python.exe")
    print("Restarting script in virtual environment...")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit(0)

# --- Imports ---
from flask import Flask, request, jsonify, render_template_string, Response
import json
import time
import base64
import threading
import re
import io
import importlib.util
from contextlib import redirect_stdout
from ollama import chat  # Ollama Python library for interfacing with models

# --- Global Variables and Directories ---
WEBCAM_DIR = "webcam_frames"
if not os.path.exists(WEBCAM_DIR):
    os.makedirs(WEBCAM_DIR)

state_lock = threading.Lock()
system_state = {
    "webcam_inference": "",
    "internal_logs": [],
    "latest_webcam_embedding": None,
}

# --- Dummy Implementations for Additional Functionalities ---
def excitatory_model(prompt):
    """
    Dummy implementation for image description.
    In production, this would call a model to describe the image.
    """
    return "This is a dummy description of the uploaded image."

def embed_text(text):
    """
    Dummy implementation to simulate text embedding.
    """
    return [0.1, 0.2, 0.3]

def store_memory(embedding, mem_type="frame", metadata=""):
    """
    Dummy memory storage function.
    """
    print(f"Storing memory of type '{mem_type}' with metadata: {metadata}")

# --- Deepseek Helper Function ---
def extract_think(text):
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start != -1 and end != -1 and end > start:
        return text[start + len(start_tag):end].strip()
    return text.strip()

# --- Install Missing Imports Helper ---
def install_missing_imports(code):
    """
    Scan the provided code for import statements and install missing packages.
    """
    lines = code.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("import "):
            parts = line.split()
            if len(parts) >= 2:
                module = parts[1].split('.')[0]
                if not importlib.util.find_spec(module):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                    except Exception as e:
                        print(f"Error installing {module}: {e}")
        elif line.startswith("from "):
            parts = line.split()
            if len(parts) >= 2:
                module = parts[1].split('.')[0]
                if not importlib.util.find_spec(module):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                    except Exception as e:
                        print(f"Error installing {module}: {e}")

# --- Updated Tool Call Extraction Helper with Iterative Error Handling ---
def extract_tool_call(text):
    """
    Extracts a function call block (wrapped in either ```tool_code or ```python)
    from the text, installs missing imports if needed, and attempts to execute the code
    iteratively until a result is produced that does not include an error.
    The function builds an iteration log and returns the final output wrapped in
    <pre><code class="hljs">...</code></pre> for syntax highlighting.
    """
    pattern = r"```(?:tool_code|python)\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        install_missing_imports(code)
        iteration_log = ""
        attempts = 0
        max_attempts = 3
        output = ""
        while attempts < max_attempts:
            attempts += 1
            iteration_log += f"Iteration {attempts}: Executing code...\n"
            f = io.StringIO()
            try:
                with redirect_stdout(f):
                    exec(code, globals())
                output = f.getvalue()
                if "Error" in output or output.strip() == "":
                    iteration_log += f"Iteration {attempts} produced an error or empty output.\n"
                else:
                    iteration_log += f"Iteration {attempts} successful.\n"
                    break
            except Exception as e:
                iteration_log += f"Iteration {attempts} raised an exception: {str(e)}\n"
                output = f"Error: {str(e)}"
        final_message = iteration_log + "\nFinal Output:\n" + output
        return f"<pre><code class='hljs'>{final_message}</code></pre>"
    return None

# --- New Function: Web Search Using Chrome ---
def search_web(query: str) -> str:
    """
    Open a Chrome browser instance (GUI mode) to perform a Google search for the given query.
    Iterates over the top 3 search results, extracts page text, and returns the integrated content.
    
    Requires Selenium and chromedriver-autoinstaller to be installed.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import json
        import chromedriver_autoinstaller
        
        # Automatically install/update chromedriver
        chromedriver_autoinstaller.install()
        
        options = Options()
        # For GUI mode, comment out the headless option.
        # options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        
        # Wait and enter the query
        search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "q")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.submit()
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")
        content = ""
        for result in results[:3]:
            try:
                link = result.find_element(By.TAG_NAME, "a")
                url = link.get_attribute("href")
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                page_content = driver.find_element(By.TAG_NAME, "body").text
                content += f"\n---\n{page_content}\n---\n"
                driver.back()
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
            except Exception as e:
                continue
        driver.quit()
        return content
    except Exception as e:
        return f"Error during web search: {str(e)}"

# --- Example Function: IP Location Checker ---
def get_ip_location(ip: str = None) -> dict:
    """
    Get the location details of your public IP address using the ipconfig.io service.
    
    The function accepts an optional 'ip' parameter, but it is ignored.
    It always retrieves your public IP automatically via ipconfig.io.
    
    Returns:
      A dictionary containing location information (such as city, region, country, etc.).
      In case of an error, returns a dict with an "error" key.
    """
    import requests
    try:
        response = requests.get("https://ipconfig.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}

# --- Example Function: Current Datetime ---
def get_current_datetime() -> str:
    """
    Return the current date and time as a formatted string.
    
    Returns:
      A string representing the current date and time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Flask App Setup ---
app = Flask(__name__)

# --- HTML Frontend Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gemma3:12b Function Calling Interface</title>
  <!-- Doto Font -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Doto:wght@100..900&display=swap" rel="stylesheet">
  <!-- Highlight.js CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>hljs.highlightAll();</script>
  <!-- Marked.js for markdown formatting -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    /* Universal Box Sizing and Body Flex Setup */
    html, body {
      box-sizing: border-box;
      height: 100vh;
      display: flex;
      flex-direction: column;
      margin: 0;
      padding: 1rem;
      gap: 1rem;
      background: black;
      color: white;
      font-family: 'Doto', sans-serif;
    }
    html { padding: 0; }
    textarea { box-sizing: border-box; }
    /* Utility classes */
    .br-1 { border-radius: 1rem; }
    
    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0rem 1rem;
      background-color: #1e1e1e;
      border-radius: 1rem;
    }
    .header button { margin-right: 1rem; }
    
    /* Container Layout */
    .container {
      flex: 1;
      display: flex;
      gap: 1rem;
    }
    
    /* Sidebar */
    .sidebar {
      width: 300px;
      background-color: #1e1e1e;
      padding: 1rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      border-radius: 1rem;
      max-height: calc(100vh - 150px);
    }
    
    /* Global Functions Section */
    .global-functions {
      padding: 0.5rem;
      border: 1px solid #444;
      border-radius: 1rem;
      background-color: #2a2a2a;
    }
    .global-functions h3 { margin: 0 0 0.5rem 0; }
    .global-functions textarea {
      width: 100%;
      height: 5rem;
      margin-bottom: 0.5rem;
      border-radius: 1rem;
    }
    .global-functions button {
      width: 100%;
      margin-bottom: 0.5rem;
      border-radius: 1rem;
    }
    .global-functions-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      overflow: auto;
    }
    .global-function-item {
      position: relative;
      background-color: #333;
      border: 1px solid #555;
      padding: 0.5rem;
      border-radius: 1rem;
      font-family: monospace;
      white-space: pre-wrap;
      font-size: 0.9rem;
    }
    .global-function-item button.remove {
      position: absolute;
      top: 5px;
      right: 5px;
      background: transparent;
      border: none;
      color: #e0e0e0;
      cursor: pointer;
    }
    .global-function-item label {
      display: block;
      margin-top: 0.5rem;
      font-size: 0.8rem;
    }
    pre code {
      display: block;
      overflow-x: auto;
      padding: 0.5rem;
      background: #1e1e1e;
      border-radius: 1rem;
      max-width:100%;
    }
    
    /* Main Section */
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: calc(100vh - 7.5rem);
    }
    .chat-history {
      flex: 1;
      padding: 10px;
      overflow-y: auto;
      border: 1px solid #333;
      border-radius: 1rem;
    }
    .flex-row { display: flex; gap: 1rem; }
    .flex-col {
      flex: 1;
      gap: 1rem;
      display: flex;
      flex-direction: column;
    }
    .flex-col > textarea { height: 100%; }
    input, textarea {
      width: 100%;
      padding: 8px;
      background-color: #2a2a2a;
      color: #e0e0e0;
      border: 1px solid #444;
      border-radius: 1rem;
      box-sizing: border-box;
    }
    button {
      padding: 10px;
      background-color: #3a3a3a;
      color: #e0e0e0;
      border: none;
      cursor: pointer;
      border-radius: 1rem;
    }
    button:hover { background-color: #555; }
    #webcamVideo { width: 100%; border-radius: 1rem; }
    
    @media (max-width: 768px) {
      .container { flex-direction: column; }
      .sidebar { width: 100%; height: auto; }
      .main { height: auto; }
    }
    
    /* Doto Font Custom Class Example */
    .doto-custom {
      font-family: 'Doto', sans-serif;
      font-optical-sizing: auto;
      font-weight: 400;
      font-style: normal;
      font-variation-settings: 'ROND' 0;
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <button onclick="toggleStorage()">Toggle Save History</button>
      <button onclick="clearHistory()">Clear History</button>
      <button onclick="downloadHistory()">Download History</button>
      <button id="toggleStreamBtn" onclick="toggleStreaming()">Streaming: ON</button>
      <button onclick="toggleCameraStream()">Toggle Video</button>
    </div>
    <h2 class="doto-custom">Gemma3:12b Chat Interface</h2>
  </div>
  <div class="container">
    <div class="sidebar">
      <h3>Parameters</h3>
      <label>Model:</label>
      <input type="text" id="model" value="gemma3:12b">
      <label>Temperature:</label>
      <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="1">
      <label>Max Tokens:</label>
      <input type="number" id="max_tokens" value="128000">
      <div class="global-functions">
        <h3>Global Functions</h3>
        <textarea id="globalFunctionInput" placeholder="Enter global function code"></textarea>
        <button onclick="saveGlobalFunction()">Save Function</button>
        <div class="global-functions-list" id="globalFunctionsList"></div>
      </div>
    </div>
    <div class="main">
      <div class="chat-history" id="chatHistory"></div>
      <div class="flex-row">
        <div class="flex-col">
          <textarea id="chatInput" placeholder="Enter your message"></textarea>
        </div>
        <div class="flex-col">
          <textarea id="functionInput" placeholder="Enter function code (optional)"></textarea>
        </div>
        <div class="flex-col">
          <video id="webcamVideo" muted></video>
          <button onclick="captureWebcam()">Capture Frame</button>
        </div>
      </div>
      <button onclick="sendPrompt()">Send Prompt</button>
    </div>
  </div>
  <script>
    // Load persistent settings from localStorage (defaults: streaming true, saveHistory true)
    let streamingMode = JSON.parse(localStorage.getItem("streamingMode") || "true");
    let saveHistory = JSON.parse(localStorage.getItem("saveHistory") || "true");
    let captureLive = JSON.parse(localStorage.getItem("captureLive") || "true");
    document.getElementById("toggleStreamBtn").innerText = streamingMode ? "Streaming: ON" : "Streaming: OFF";
    
    // Function to render markdown using Marked and post-process code block classes.
    function renderMarkdown(text) {
      if(window.marked) {
        let html = marked.parse(text);
        // Replace language identifiers "tool_code" with "python"
        html = html.replace(/language-tool_code/g, "language-python");
        return html;
      }
      return text;
    }
    
    // Function to re-apply syntax highlighting using Highlight.js
    function highlightCode() {
      document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    }
    
    function toggleStreaming() {
      streamingMode = !streamingMode;
      localStorage.setItem("streamingMode", JSON.stringify(streamingMode));
      document.getElementById("toggleStreamBtn").innerText = streamingMode ? "Streaming: ON" : "Streaming: OFF";
    }

    function toggleCameraStream() {
      captureLive = !captureLive;
      localStorage.setItem("captureLive", JSON.stringify(captureLive));
      alert("Video Capture is now " + (captureLive ? "ON" : "OFF"));
    }

    function toggleStorage() {
      saveHistory = !saveHistory;
      localStorage.setItem("saveHistory", JSON.stringify(saveHistory));
      alert("Save History is now " + (saveHistory ? "ON" : "OFF"));
    }
    function clearHistory() {
      localStorage.removeItem("chatHistory");
      document.getElementById("chatHistory").innerHTML = "";
    }
    function downloadHistory() {
      let history = localStorage.getItem("chatHistory") || "";
      let blob = new Blob([history], {type: "text/plain"});
      let url = URL.createObjectURL(blob);
      let a = document.createElement("a");
      a.href = url;
      a.download = "chat_history.txt";
      a.click();
    }
    
    // Chat History Management
    function appendChat(message) {
      let chatHistory = document.getElementById("chatHistory");
      chatHistory.innerHTML += "<div>" + renderMarkdown(message) + "</div>";
      if (saveHistory) {
        let stored = localStorage.getItem("chatHistory") || "";
        localStorage.setItem("chatHistory", stored + message + "");
      }
      chatHistory.scrollTop = chatHistory.scrollHeight;
      highlightCode();
    }
    
    // Global Functions Management
    function saveGlobalFunction() {
      const input = document.getElementById("globalFunctionInput");
      const code = input.value.trim();
      if (!code) return;
      let globalFunctions = JSON.parse(localStorage.getItem("globalFunctions") || "[]");
      globalFunctions.push({ code: code, active: true });
      localStorage.setItem("globalFunctions", JSON.stringify(globalFunctions));
      input.value = "";
      renderGlobalFunctions();
    }
    
    function removeGlobalFunction(index) {
      let globalFunctions = JSON.parse(localStorage.getItem("globalFunctions") || "[]");
      globalFunctions.splice(index, 1);
      localStorage.setItem("globalFunctions", JSON.stringify(globalFunctions));
      renderGlobalFunctions();
    }
    
    function toggleGlobalFunction(index, checkbox) {
      let globalFunctions = JSON.parse(localStorage.getItem("globalFunctions") || "[]");
      globalFunctions[index].active = checkbox.checked;
      localStorage.setItem("globalFunctions", JSON.stringify(globalFunctions));
    }
    
    function renderGlobalFunctions() {
      let container = document.getElementById("globalFunctionsList");
      let globalFunctions = JSON.parse(localStorage.getItem("globalFunctions") || "[]");
      container.innerHTML = "";
      globalFunctions.forEach((func, index) => {
        const div = document.createElement("div");
        div.className = "global-function-item";
        div.innerHTML = "<code>" + func.code + "</code>";
        const removeBtn = document.createElement("button");
        removeBtn.className = "remove";
        removeBtn.textContent = "x";
        removeBtn.onclick = () => removeGlobalFunction(index);
        div.appendChild(removeBtn);
        const label = document.createElement("label");
        label.innerHTML = "<input type='checkbox' " + (func.active ? "checked" : "") + " onchange='toggleGlobalFunction(" + index + ", this)'> Active";
        div.appendChild(label);
        container.appendChild(div);
      });
      highlightCode();
    }
    renderGlobalFunctions();
    
    // AJAX Prompt Send
    async function sendPrompt() {
      let chatText = document.getElementById("chatInput").value;
      let functionText = document.getElementById("functionInput").value;
      let imageData = window.capturedImage || "";
      let globalFunctions = JSON.parse(localStorage.getItem("globalFunctions") || "[]");
      let activeFuncs = [];
      globalFunctions.forEach(func => {
        if (func.active) {
          activeFuncs.push(func.code);
        }
      });
      if (activeFuncs.length > 0) {
        const newline = String.fromCharCode(10);
        functionText = activeFuncs.join(newline) + newline + functionText;
      }
      let params = {
        chat_text: chatText,
        function_text: functionText,
        image_data: imageData,
        model: document.getElementById("model").value,
        temperature: document.getElementById("temperature").value,
        max_tokens: document.getElementById("max_tokens").value,
        stream: streamingMode
      };
      appendChat("<b>User:</b> " + chatText);
      
      if (streamingMode) {
        const response = await fetch("/send", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(params)
        });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let result = "";
        let tempMsg = document.createElement("div");
        tempMsg.innerHTML = "<b>Model:</b> ";
        document.getElementById("chatHistory").appendChild(tempMsg);
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, {stream: true});
          result += chunk;
          tempMsg.innerHTML = "<b>Model:</b> " + renderMarkdown(result);
          highlightCode();
        }
      } else {
        const response = await fetch("/send", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(params)
        });
        const data = await response.json();
        appendChat("<b>Model:</b> " + renderMarkdown(data.response));
      }
      document.getElementById("chatInput").value = "";
      document.getElementById("functionInput").value = "";
      window.capturedImage = "";
    }
    
   function captureWebcam() {
     if (!captureLive) {
       alert("Video Capture is OFF.  Cannot capture a frame.");
       return;
     }

     let video = document.getElementById("webcamVideo");
     let canvas = document.createElement("canvas");
     canvas.width = video.videoWidth;
     canvas.height = video.videoHeight;
     let ctx = canvas.getContext("2d");
     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
     window.capturedImage = canvas.toDataURL("image/jpeg");
     alert("Webcam frame captured!");
   }

   function startCameraStream() {
     if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
       navigator.mediaDevices.getUserMedia({ video: true })
         .then(function(stream) {
           let video = document.getElementById("webcamVideo");
           video.srcObject = stream;
           video.play();
          })
         .catch(function(error) {
           console.error("Error accessing webcam:", error);
           alert("Error accessing webcam.  Please check permissions and try again.");
         });
     } else {
       alert("Webcam functionality not supported in your browser.");
     }
   }

   function stopCameraStream() {
     let video = document.getElementById("webcamVideo");
     if (video.srcObject) {
       video.srcObject.getTracks().forEach(track => track.stop());
       video.srcObject = null;
     }
   }


   function toggleCameraStream() {
     captureLive = !captureLive;
     localStorage.setItem("captureLive", JSON.stringify(captureLive));
     alert("Video Capture is now " + (captureLive ? "ON" : "OFF"));

     if (captureLive) {
       startCameraStream();
     } else {
       stopCameraStream();
     }
   }
  </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/send", methods=["POST"])
def send_prompt():
    data = request.get_json()
    chat_text = data.get("chat_text", "")
    function_text = data.get("function_text", "")
    image_data = data.get("image_data", "")
    model = data.get("model", "gemma3:12b")
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 1024))
    
    prompt = chat_text
    if function_text:
        prompt += "\nFunction Info:\n" + function_text
    if image_data:
        try:
            header, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            filename = f"{WEBCAM_DIR}/frame_{int(time.time())}.jpg"
            with open(filename, "wb") as f:
                f.write(img_bytes)
            prompt += f"\n[Webcam Image: ./{filename}]"
        except Exception as e:
            print("Error processing image:", e)
    
    if data.get("stream"):
        def generate():
            stream_response = chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
            full_response = ""
            for chunk in stream_response:
                token = chunk['message']['content']
                full_response += token
                yield token
            # After streaming, check for function call extraction
            tool_output = extract_tool_call(full_response)
            if tool_output:
                messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': full_response},
                    {'role': 'user', 'content': tool_output}
                ]
                try:
                    secondary_obj = chat(model=model, messages=messages)
                    final_response_text = secondary_obj['message']['content']
                    yield "\n" + final_response_text
                except Exception as e:
                    yield "\nError in secondary inference: " + str(e)
        return Response(generate(), mimetype='text/plain')
    else:
        try:
            response_obj = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            response_text = response_obj['message']['content']
        except Exception as e:
            response_text = f"Error calling model: {str(e)}"
    
        tool_output = extract_tool_call(response_text)
        if tool_output:
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': response_text},
                {'role': 'user', 'content': tool_output}
            ]
            try:
                secondary_obj = chat(model=model, messages=messages)
                final_response_text = secondary_obj['message']['content']
            except Exception as e:
                final_response_text = f"Error in secondary inference: {str(e)}"
        else:
            final_response_text = response_text
    
        final_output = extract_think(final_response_text)
        return jsonify({"response": final_output})

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    data = request.get_json()
    if not data or "image_data" not in data:
        return "No image data provided", 400
    image_data = data["image_data"]
    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        filename = f"{WEBCAM_DIR}/frame_{int(time.time())}.jpg"
        with open(filename, "wb") as f:
            f.write(img_bytes)
        prompt = f"describe this image: ./{filename}"
        webcam_result = excitatory_model(prompt)
        embedding = embed_text(webcam_result)
        store_memory(embedding, mem_type="frame", metadata=f"Webcam frame described: {webcam_result}")
        with state_lock:
            system_state["webcam_inference"] = webcam_result
            system_state["internal_logs"].append(f"Webcam Inference: {webcam_result}")
            system_state["latest_webcam_embedding"] = embedding
        return json.dumps({"result": webcam_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
