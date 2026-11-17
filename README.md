# VoxLibro Local — Private AI PDF Reader & Audiobook Web App

VoxLibro Local is an offline-first, zero-cost, private AI-powered PDF audiobook reader engineered specifically around real iOS and Safari constraints.

## 1. Honest iOS & Safari Technical Architecture (Phase 1 Validation)

### The Core iOS Constraint: Background Audio Playback vs. Background AI Synthesis
On iOS (Safari and Home-Screen Web Apps / PWAs), WebKit strictly distinguishes between:
1. **Background Audio Playback (Supported)**: An active HTML5 `<audio>` element playing a media stream or Blob URL continues playing when the user locks the iPhone screen or switches apps, and integrates with the iOS Lock Screen and Control Center via the [MDN Media Session API](https://developer.mozilla.org/en-US/docs/Web/API/Media_Session_API) (`navigator.mediaSession`).
2. **Background JavaScript / WebAssembly / WebGPU Compute (Suspended by iOS)**: Within 5–30 seconds of locking the iPhone screen, iOS suspends background `setTimeout` loops, `WebWorker` inference threads, and `speechSynthesis.speak()` transitions between utterances. If an app relies on firing a JavaScript `onended` callback after every single sentence to synthesize the next sentence while the screen is locked, playback stops after the current sentence finishes.

### How VoxLibro Solves iOS Locked-Screen Playback
Instead of synthesizing one sentence at a time on a foreground JS timer:
- **Lookahead Concatenated Audio Windowing**: VoxLibro pre-generates a rolling window of upcoming speech chunks (e.g., 10–30 paragraphs = 15–45 minutes of continuous speech) using the local neural TTS engine ([Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) or [Piper TTS WASM](https://github.com/Mintplex-Labs/piper-tts-web)) and stitches their PCM samples into a single continuous `.wav` stream with exact chunk-timestamp boundaries (`chunkTimeline[]`).
- **Single Continuous `<audio>` Element**: That stitched `.wav` Blob is loaded into a single persistent `<audio>` element with `preservesPitch = true` (`webkitPreservesPitch = true`).
- **Lock-Screen Timecode Tracking**: While the iPhone screen is locked, iOS natively plays the entire stitched multi-paragraph `.wav` stream without needing foreground JS inference. As `timeupdate` events fire (or when the user unlocks the screen or taps Lock-Screen Play/Pause/Next/Prev via `navigator.mediaSession.setActionHandler`), VoxLibro maps `audioElement.currentTime` back to the exact `chunkIndex`, `pageNumber`, `paragraph`, `sentence`, and `charOffset` in IndexedDB / SQLite.

## 2. Three Zero-Cost Installation & Hosting Pathways on iPhone

No paid Apple Developer account ($99/yr), paid cloud hosting, or external LLM/TTS API key is required:

### Option A: 100% On-Device Standalone Home-Screen Web App (Zero Desktop Required)
1. Open `private_ai_pdf_audiobook.html` in Safari on your iPhone (or serve the static `frontend/dist` folder once over local Wi-Fi so the Service Worker `sw.js` caches all assets and ONNX/WASM weights into IndexedDB/OPFS).
2. Tap the **Share** button in Safari and select **Add to Home Screen**.
3. Launch **VoxLibro AI** from your iPhone Home Screen. All imported PDFs, generated `.wav` audio caches, reading positions, and AI notes persist locally in [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) + `localStorage`.

### Option B: On-Device Local Python Backend on iPhone (a-Shell / iSH) or Companion LAN Server
1. Install the free open-source **a-Shell** app from the iOS App Store (runs Python 3 natively on iPhone localhost) or run the backend on your personal laptop over local Wi-Fi:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. Place quantized GGUF models (such as `gemma-4-E4B-it-Q4_K_M.gguf` or `Llama-3.2-1B-Instruct-Q4_K_M.gguf`) and `kokoro-v1.0.onnx` / `en_US-lessac-high.onnx` inside `backend/local_storage/models/`.

### Option C: Free Native iOS Wrapper via Capacitor (Optional, Requires Mac with Free Personal Apple ID)
If you want native `AVAudioSessionCategoryPlayback` background execution:
```bash
npx cap add ios
# In ios/App/App/Info.plist, include UIBackgroundModes -> audio
```
Build and sideload to your iPhone using a free Personal Team Apple ID in Xcode or via AltStore/SideStore ($0 cost).
