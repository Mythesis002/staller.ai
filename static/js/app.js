(function(){
  const uploadedPaths = []; // items: { path, url, name }

  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput");
  const uploadBtn = $("uploadBtn");
  const attachBtn = $("attachBtn");
  const attachmentsTray = $("attachmentsTray");
  const uploadedList = $("uploadedList");
  const promptEl = $("prompt");
  const sendBtn = $("sendBtn");
  const contentEl = $("content");
  const analysesEl = $("analyses");
  const jsonPlanEl = $("jsonPlan");
  const planIdDisplay = $("planIdDisplay");
  const videoPlayer = $("videoPlayer");
  const renderInfo = $("renderInfo");
  const chatEl = document.getElementById("chat");
  // Keep the most recent plan/refine result to populate thinking details
  let lastPlanResult = null;
  let isProcessing = false;

  function setProcessing(on){
    isProcessing = !!on;
    try{
      if (sendBtn) sendBtn.disabled = on;
      if (attachBtn) attachBtn.disabled = on;
      if (fileInput) fileInput.disabled = on;
      if (promptEl) promptEl.disabled = on;
      const composer = document.querySelector('.composer-inner');
      if (composer) composer.classList.toggle('busy', on);
    }catch(e){}
  }

  function showUploaded() {
    if (uploadedList) uploadedList.textContent = uploadedPaths.map(i=>i.name||i.path).join("\n");
    // Render thumbnails in composer tray with remove buttons
    if (attachmentsTray) {
      attachmentsTray.innerHTML = renderComposerAttachments(uploadedPaths);
    }
  }

  function buildThinkingToggle(result){
    if (!result) return "";
    const id = `think_${Math.random().toString(36).slice(2)}`;
    const details = buildThinkingDetails(result);
    if (!details) return "";
    return `<button class=\"toggleJson\" data-target=\"${id}\">Show thinking</button>
            <div id=\"${id}\" class=\"thinking-details\" style=\"display:none; margin-top:6px;\">${details}</div>`;
  }

  function buildThinkingDetails(result){
    try{
      const parts = [];
      if (result.analyses && typeof result.analyses === 'object'){
        const items = Object.entries(result.analyses).map(([file, a]) => {
          const ok = a && (a.status === 'success' || a.status === 'ok');
          const ft = a && a.file_type ? a.file_type : 'unknown';
          let preview = '';
          const text = a && (a.analysis || a.summary || a.text);
          if (text) preview = String(text).slice(0,150) + (String(text).length>150?'…':'');
          return `<div>${ok?'✅':'⚠️'} <strong>${escapeHtml(file)}</strong> (${escapeHtml(ft)})${preview?` — ${escapeHtml(preview)}`:''}</div>`;
        }).join("");
        if (items) parts.push(`<div><strong>Media analysis</strong>${items}</div>`);
      }
      const tech = [];
      if (result.plan_id) tech.push(`<div>Plan ID: ${escapeHtml(result.plan_id)}</div>`);
      if (result.render_id) tech.push(`<div>Render ID: ${escapeHtml(result.render_id)}</div>`);
      if (result.render_status) tech.push(`<div>Status: ${escapeHtml(String(result.render_status))}</div>`);
      if (result.video_url) tech.push(`<div>Video URL: ${escapeHtml(result.video_url)}</div>`);
      if (tech.length) parts.push(`<div style=\"margin-top:6px;\"><strong>Technical</strong>${tech.join('')}</div>`);
      const html = parts.join('');
      return html || "";
    }catch{ return ""; }
  }

  // Toggle handler (delegated on chat container) — global scope
  chatEl.addEventListener('click', (e) => {
    const btn = e.target.closest('button.toggleJson');
    if (!btn) return;
    const target = btn.getAttribute('data-target');
    if (!target) return;
    const el = document.getElementById(target);
    if (!el) return;
    const vis = el.style.display !== 'none';
    el.style.display = vis ? 'none' : 'block';
    btn.textContent = vis ? 'Show thinking' : 'Hide thinking';
  });

  function appendChat(role, html) {
    const wrap = document.createElement("div");
    wrap.className = `chat-msg ${role}`;
    const label = `<div class="sender-label">${role === 'user' ? 'You' : 'Staller'}</div>`;
    wrap.innerHTML = label + html;
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
    return wrap;
  }

  function startThinking(label){
    const id = `think_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const el = appendChat("assistant", `<div class=msg id='${id}'><div class='thinking'><span class='dot'></span><span class='dot'></span><span class='dot'></span></div><div class='thinking-label'>${escapeHtml(label||"Thinking…")}</div></div>`);
    return {
      id,
      set(text){ if (el) el.innerHTML = `<div class=msg><div class='thinking'><span class='dot'></span><span class='dot'></span><span class='dot'></span></div><div class='thinking-label'>${escapeHtml(text||"…")}</div></div>`; },
      replace(html){ if (el) el.innerHTML = html; },
      error(message){ if (el) el.innerHTML = `<div class=msg><strong>Error</strong><div>${escapeHtml(message||"Request failed")}</div></div>`; }
    };
  }

  async function uploadSelectedFiles() {
    const files = fileInput.files;
    if (!files || files.length === 0) return;
    if (isProcessing) return; // avoid overlap
    setProcessing(true);
    // Track per-file parallel state so we only remove overlay when both cloud and analysis are done
    const statusByName = new Map(); // name -> { cloudDone:boolean, analyzeDone:boolean }
    const fileList = Array.from(files);

    // Create provisional chips with overlay in the attachments tray
    const tempChips = new Map();
    function addTempChip(file){
      const url = (window.URL || window.webkitURL).createObjectURL(file);
      const n = file.name || '';
      const isVid = isVideo(n);
      const isAud = isAudio(n);
      const media = isVid
        ? `<video src="${url}" muted playsinline></video>`
        : isAud
          ? `<audio src="${url}" preload="metadata" controls></audio>`
          : `<img src="${url}" alt="${escapeHtml(n)}"/>`;
      const chip = document.createElement('div');
      chip.className = 'att att-temp';
      chip.setAttribute('data-temp', file.name);
      chip.innerHTML = `${media}
        <div class="overlay">
          <div class="ov-inner">
            <div class="spinner"></div>
            <div class="bars">
              <div class="bar local"><span style="width:0%"></span></div>
              <div class="bar cloud"><span style="width:0%"></span></div>
            </div>
            <div class="chip">Uploading…</div>
          </div>
        </div>`;
      attachmentsTray && attachmentsTray.appendChild(chip);
      tempChips.set(file.name, chip);
    }
    function setLocalPct(name, pct){ const el = tempChips.get(name); if (!el) return; const bar = el.querySelector('.bar.local > span'); if (bar) bar.style.width = (pct||0)+'%'; const chip = el.querySelector('.chip'); if (chip) chip.textContent = `Local ${pct|0}%`; }
    function setCloudPct(name, pct){ const el = tempChips.get(name); if (!el) return; const bar = el.querySelector('.bar.cloud > span'); if (bar) bar.style.width = (pct||0)+'%'; const chip = el.querySelector('.chip'); if (chip) chip.textContent = `Cloud ${pct|0}%`; }
    function setAnalyzeState(name, text){ const el = tempChips.get(name); if (!el) return; const chip = el.querySelector('.chip'); if (chip) chip.textContent = text; }
    function maybeCompleteChip(name){
      const st = statusByName.get(name) || {};
      if (st.cloudDone && st.analyzeDone){
        const el = tempChips.get(name); if (!el) return; const ov = el.querySelector('.overlay'); if (ov) ov.remove();
      }
    }
    fileList.forEach(addTempChip);
    try {
      // No local upload anymore

      // Helper: direct Cloudinary upload with progress (signed)
      async function uploadCloudinaryXHR(file){
        // get signature
        const sigRes = await fetch('/media/cloudinary/sign');
        if (!sigRes.ok) throw new Error('sign failed');
        const sig = await sigRes.json();
        const url = `https://api.cloudinary.com/v1_1/${encodeURIComponent(sig.cloud_name)}/auto/upload`;
        return new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open('POST', url);
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable){
              const pct = Math.round((e.loaded / e.total) * 100);
              setCloudPct(file.name, pct);
            }
          };
          xhr.onreadystatechange = () => {
            if (xhr.readyState === 4){
              if (xhr.status >=200 && xhr.status<300){
                try { resolve(JSON.parse(xhr.responseText||'{}')); } catch(e){ resolve({}); }
              } else { reject(new Error('cloud upload failed')); }
            }
          };
          const fd = new FormData();
          fd.append('file', file);
          fd.append('api_key', sig.api_key);
          fd.append('timestamp', String(sig.timestamp));
          fd.append('signature', sig.signature);
          if (sig.folder) fd.append('folder', sig.folder);
          xhr.send(fd);
        });
      }

      // Cloud-only: mark local as 100% immediately for UI clarity
      fileList.forEach(f => { setLocalPct(f.name, 100); statusByName.set(f.name, { cloudDone: false, analyzeDone: false }); });

      const cloudPromises = fileList.map(async (f) => {
        try {
          const c = await uploadCloudinaryXHR(f);
          const secureUrl = c && (c.secure_url || c.url);
          if (secureUrl) {
            setCloudPct(f.name, 100);
            // Track in uploaded list
            uploadedPaths.push({ url: secureUrl, name: f.name });
            // Immediately analyze via remote endpoint
            try {
              setAnalyzeState(f.name, 'Analyzing…');
              const resA = await fetch('/media/analyze_remote', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ url: secureUrl, filename: f.name }) });
              if (resA.ok){
                const a = await resA.json();
                console.log('Remote analysis result:', a);
                const idx = uploadedPaths.findIndex(it => it.url === secureUrl);
                if (idx >= 0) uploadedPaths[idx].analyzed = true;
                setAnalyzeState(f.name, 'Ready');
              } else {
                setAnalyzeState(f.name, 'Analyze failed');
              }
            } catch {
              setAnalyzeState(f.name, 'Analyze error');
            } finally {
              const st = statusByName.get(f.name) || {}; st.analyzeDone = true; statusByName.set(f.name, st); maybeCompleteChip(f.name);
            }
          }
        } catch {
          // ignore; UI already showed progress
        } finally {
          const st = statusByName.get(f.name) || {}; st.cloudDone = true; statusByName.set(f.name, st); maybeCompleteChip(f.name);
        }
      });

      await Promise.allSettled([...cloudPromises]);
      // Re-render chips without overlays only if both tasks finished per file (handled by maybeCompleteChip)
      showUploaded();
    } finally {
      fileInput.value = "";
      setProcessing(false);
    }
  }

  // Legacy hidden button handler kept
  uploadBtn.addEventListener("click", uploadSelectedFiles);

  // New composer: clicking attach opens picker; auto-upload on selection
  if (attachBtn) {
    attachBtn.addEventListener("click", () => { if (isProcessing) return; fileInput && fileInput.click(); });
  }
  if (fileInput) {
    fileInput.addEventListener("change", uploadSelectedFiles);
  }

  // Remove attachment chip (event delegation)
  if (attachmentsTray) {
    attachmentsTray.addEventListener('click', (e) => {
      const btn = e.target.closest('.remove');
      if (!btn) return;
      if (isProcessing) return;
      const u = btn.getAttribute('data-url');
      if (!u) return;
      const idx = uploadedPaths.findIndex(it => it.url === u);
      if (idx >= 0) {
        uploadedPaths.splice(idx, 1);
        showUploaded();
      }
    });
  }

  // Track which files were used in the initial plan so we can compute "new" ones for refinement
  let initialPlanMediaSnapshot = null;
  let currentPlanId = "";

  // Enter to send (Shift+Enter for newline)
  if (promptEl) {
    promptEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
      }
    });
  }

  sendBtn.addEventListener("click", async () => {
    const prompt = promptEl.value || "";
    if (!prompt) return alert("Please enter a prompt.");
    if (isProcessing) return;
    setProcessing(true);
    // clear input for professional feel
    promptEl.value = "";

    if (!currentPlanId) {
      // First send -> create plan
      appendChat("user", `<div class=msg><div class=prompt>${escapeHtml(prompt)}</div>${renderAttachments(uploadedPaths)}</div>`);
      startPlanStream(prompt, uploadedPaths.map(i=>i.url));
    } else {
      // Subsequent sends -> refinement; treat any newly uploaded paths as new_media_files
      const newMedia = uploadedPaths.filter(p => !(initialPlanMediaSnapshot||[]).some(x=>x.url===p.url));
      appendChat("user", `<div class=msg><div class=prompt>${escapeHtml(prompt)}</div>${renderAttachments(newMedia)}</div>`);
      startRefineStream(currentPlanId, prompt, newMedia.map(i=>i.url));
    }
  });

  function startPlanStream(prompt, mediaUrls){
    // If every attached media has been pre-analyzed, start directly at "Planning…"
    const allAnalyzed = (mediaUrls && mediaUrls.length)
      ? mediaUrls.every(u => {
          const it = uploadedPaths.find(x => x.url === u);
          return it && it.analyzed === true;
        })
      : false;
    const thinking = startThinking(allAnalyzed ? "Planning…" : (mediaUrls && mediaUrls.length ? "Analyzing media…" : "Planning…"));
    const urlsParam = encodeURIComponent((mediaUrls||[]).join('|'));
    const url = `/stream/plan?prompt=${encodeURIComponent(prompt)}&media_urls=${urlsParam}&skip_media=${allAnalyzed?1:0}`;
    streamWithEvents(url, thinking, (payload) => {
      lastPlanResult = payload;
      currentPlanId = payload.plan_id || "";
      if (planIdDisplay) planIdDisplay.textContent = currentPlanId || "none";
      initialPlanMediaSnapshot = [...uploadedPaths];
      if (contentEl) contentEl.textContent = payload.content ? String(payload.content) : "";
      if (analysesEl) analysesEl.textContent = payload.analyses ? JSON.stringify(payload.analyses, null, 2) : "";
      if (jsonPlanEl) jsonPlanEl.textContent = payload.json_plan ? JSON.stringify(payload.json_plan, null, 2) : "";
      updateRenderUI(payload);
    }, { suppressAnalyze: allAnalyzed });
  }

  function startRefineStream(planId, prompt, newMediaUrls){
    const thinking = startThinking(newMediaUrls && newMediaUrls.length ? "Analyzing new media…" : "Refining…");
    const nm = encodeURIComponent((newMediaUrls||[]).join('|'));
    const url = `/stream/refine?plan_id=${encodeURIComponent(planId)}&prompt=${encodeURIComponent(prompt)}&new_media_urls=${nm}`;
    streamWithEvents(url, thinking, (payload) => {
      lastPlanResult = payload;
      currentPlanId = payload.plan_id || currentPlanId;
      if (planIdDisplay) planIdDisplay.textContent = currentPlanId || "none";
      if (contentEl) contentEl.textContent = payload.content ? String(payload.content) : "";
      if (analysesEl) analysesEl.textContent = payload.analyses ? JSON.stringify(payload.analyses, null, 2) : "";
      if (jsonPlanEl) jsonPlanEl.textContent = payload.json_plan ? JSON.stringify(payload.json_plan, null, 2) : "";
      if (newMediaUrls && newMediaUrls.length) initialPlanMediaSnapshot = [...uploadedPaths];
      updateRenderUI(payload);
    });
  }

  // Typing effect function
  function typeWriter(text, element, speed = 20) {
    let i = 0;
    element.textContent = '';
    function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
        setTimeout(type, speed);
      }
    }
    type();
  }

  function streamWithEvents(url, thinking, onResult, opts){
    try{
      const es = new EventSource(url);
      let directorContentElement = null;
      let progressBoxElement = null;
      
      es.onmessage = (evt) => {
        try{
          const data = JSON.parse(evt.data||'{}');
          switch(data.type){
            case 'step':
              {
                const msg = String(data.message||'');
                // If we pre-analyzed, suppress noisy 'Analyzing' step and proceed
                if (opts && opts.suppressAnalyze && /Analyzing/i.test(msg)) {
                  // ignore this step
                  return;
                }
                thinking.set(msg);
              }
              break;
            case 'director_complete':
              {
                // Director finished - show content with typing effect + progress box
                const contentText = String(data.content || '');
                const html = `<div class=msg>
                    <div class='director-content'></div>
                    <div class='edit-progress-box'>
                      <div class='progress-icon'>🎬</div>
                      <div class='progress-text'>Generating Edit...</div>
                    </div>
                  </div>`;
                thinking.replace(html);
                
                // Get references to the elements
                const msgElement = chatMessages.lastElementChild;
                if (msgElement) {
                  directorContentElement = msgElement.querySelector('.director-content');
                  progressBoxElement = msgElement.querySelector('.edit-progress-box');
                  
                  // Start typing effect
                  if (directorContentElement && contentText) {
                    typeWriter(contentText, directorContentElement, 20);
                  }
                }
              }
              break;
            case 'result':
              if (onResult) onResult(data.payload||{});
              // If immediate URL inside payload, finish; otherwise show a 16:9 placeholder with spinner
              if (data.payload && (data.payload.video_url || (data.payload.rendered_video && data.payload.rendered_video.video_url))){
                const vu = data.payload.video_url || data.payload.rendered_video.video_url;
                const html = `<div class=msg>
                    <div>${escapeHtml(data.payload.content||'Your video is ready.')}</div>
                    ${buildThinkingToggle(lastPlanResult)}
                    <div class='chat-video'><video src='${vu}' controls playsinline></video></div>
                  </div>`;
                thinking.replace(html);
                es.close();
                setProcessing(false);
              } else {
                // No URL yet: show Director content and a 16:9 blurred placeholder with spinner
                const html = `<div class=msg>
                    <div>${escapeHtml((data.payload && data.payload.content) || 'Generating your video…')}</div>
                    ${buildThinkingToggle(lastPlanResult)}
                    <div class='chat-video'>
                      <div class='video-placeholder'>
                        <div class='spinner large'></div>
                        <div class='label'>Generating video…</div>
                      </div>
                    </div>
                  </div>`;
                thinking.replace(html);
              }
              break;
            case 'render':
              {
                // Update placeholder label if present; fallback to legacy thinking.set
                const lbl = document.querySelector('.video-placeholder .label');
                const text = `Rendering… ${data.progress != null ? data.progress+"%" : (data.status||"")}`;
                if (lbl) {
                  lbl.textContent = text;
                } else {
                  thinking.set(text);
                }
              }
              break;
            case 'done':
              if (data.video_url){
                videoPlayer.src = data.video_url; videoPlayer.style.display = 'block';
                
                // If we have director content already displayed, just replace progress box with video
                if (directorContentElement && progressBoxElement) {
                  // Remove progress box
                  progressBoxElement.remove();
                  
                  // Add thinking toggle and video after content
                  const msgElement = directorContentElement.closest('.msg');
                  if (msgElement) {
                    const thinkingToggle = buildThinkingToggle(lastPlanResult);
                    const videoHtml = `${thinkingToggle}<div class='chat-video'><video src='${data.video_url}' controls playsinline></video></div>`;
                    msgElement.insertAdjacentHTML('beforeend', videoHtml);
                  }
                } else {
                  // Fallback: replace entire thinking bubble
                  const html = `<div class=msg>
                      <div>${escapeHtml((lastPlanResult && lastPlanResult.content) || 'Your video is ready.')}</div>
                      ${buildThinkingToggle(lastPlanResult)}
                      <div class='chat-video'><video src='${data.video_url}' controls playsinline></video></div>
                    </div>`;
                  thinking.replace(html);
                }
              } else {
                thinking.replace(`<div class=msg><div>${escapeHtml((lastPlanResult && lastPlanResult.content) || 'Completed.')}</div></div>`);
              }
              es.close();
              setProcessing(false);
              break;
            case 'error':
              thinking.error(String(data.message||'Stream error'));
              es.close();
              setProcessing(false);
              break;
            default:
              break;
          }
        }catch(err){ /* ignore malformed */ }
      };
      es.onerror = () => {
        thinking.error('Connection lost');
        try{ es.close(); }catch(_e){}
        setProcessing(false);
      };
    }catch(e){
      thinking.error(String(e));
      setProcessing(false);
    }
  }

  function updateRenderUI(data){
    const status = data.render_status || (data.rendered_video && data.rendered_video.status);
    const message = (data.rendered_video && data.rendered_video.message) || data.message || "";
    const url = data.video_url || (data.rendered_video && data.rendered_video.video_url);
    const rid = data.render_id || (data.rendered_video && data.rendered_video.render_id);
    renderInfo.textContent = JSON.stringify({ status, message, render_id: rid, video_url: url }, null, 2);
    if (url) {
      videoPlayer.src = url;
      videoPlayer.style.display = "block";
    } else {
      videoPlayer.removeAttribute("src");
      videoPlayer.style.display = "none";
    }
  }

  function buildAssistantHTMLText(text, title){
    return `<div class=msg><strong>${escapeHtml(title)}</strong><div>${escapeHtml(text||"")}</div></div>`;
  }
  function appendAssistantText(text, title){
    const html = buildAssistantHTMLText(text, title);
    appendChat("assistant", html);
  }

  // Upload status bubble helpers (with thumbnails and per-channel states)
  function startUploadStatus(files){
    const id = `upl_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const rows = (files||[]).map(f => {
      const url = (window.URL || window.webkitURL).createObjectURL(f);
      const isVid = /\.(mp4|mov|webm|mkv|avi)$/i.test(f.name);
      const thumb = isVid ? `<video src="${url}" muted playsinline></video>` : `<img src="${url}" alt="${escapeHtml(f.name)}"/>`;
      return `<div class="row" data-name="${escapeHtml(f.name)}">
         <span class="icon spin"></span>
         <div class="thumb">${thumb}</div>
         <div class="name">${escapeHtml(f.name)}
           <div class="state">
             <div class="ln local">Local: <span class="p">0%</span></div>
             <div class="ln cloud">Cloud: <span class="p">0%</span></div>
             <div class="ln analyze">Analyze: <span class="p">waiting</span></div>
           </div>
         </div>
       </div>`;
    }).join("");
    const html = `<div class="msg" id="${id}">
        <div class="upload-status">
          <div class="title">Preparing media…</div>
          <div class="rows">${rows}</div>
        </div>
      </div>`;
    const wrap = appendChat("assistant", html);
    const root = wrap.querySelector(`#${id}`);
    function findRow(name){ return root && root.querySelector(`.row[data-name="${CSS.escape(name)}"]`); }
    function setTitle(t){ const el = root && root.querySelector('.title'); if (el) el.textContent = t; }
    function setIcon(name, cls){ const row = findRow(name); if (!row) return; const icon = row.querySelector('.icon'); icon.className = `icon ${cls}`.trim(); }
    function setLocal(name, pct){ const row = findRow(name); if (!row) return; const el = row.querySelector('.ln.local .p'); if (el) el.textContent = (pct!=null? pct+"%" : ""); }
    function setCloud(name, pct){ const row = findRow(name); if (!row) return; const el = row.querySelector('.ln.cloud .p'); if (el) el.textContent = (pct!=null? pct+"%" : ""); }
    function setAnalyze(name, text){ const row = findRow(name); if (!row) return; const el = row.querySelector('.ln.analyze .p'); if (el) el.textContent = text || ''; }
    function setCloudLink(name, link){ const row = findRow(name); if (!row) return; let a = row.querySelector('a.cloud'); if (!a){ a = document.createElement('a'); a.className='cloud'; a.style.marginLeft='8px'; a.style.fontSize='12px'; a.style.color='#9cc5ff'; a.target='_blank'; row.querySelector('.ln.cloud')?.appendChild(a);} a.href=link; a.textContent='open'; }
    return { setTitle, setIcon, setLocal, setCloud, setAnalyze, setCloudLink };
  }

  async  function maybePollRender(data, thinkingHandle, assistantText){
    const rid = data.render_id || (data.rendered_video && data.rendered_video.render_id);
    const url = data.video_url || (data.rendered_video && data.rendered_video.video_url);
    const status = data.render_status || (data.rendered_video && data.rendered_video.status);
    if (!rid || url || (status && ["done","completed","success","validation_error","error"].includes(String(status)))) return;
    pollRender(rid, thinkingHandle, assistantText);
  }

  async function pollRender(renderId, thinkingHandle, assistantText){
    try{
      let tries = 0;
      while(tries < 60){ // up to ~3 minutes at 3s interval
        const res = await fetch(`/renders/${encodeURIComponent(renderId)}`);
        if(!res.ok) break;
        const info = await res.json();
        renderInfo.textContent = JSON.stringify({ status: info.status, progress: info.progress, video_url: info.video_url, message: info.message }, null, 2);
        if (thinkingHandle) thinkingHandle.set(`Rendering… ${info.progress != null ? info.progress+"%" : (info.status||"")}`);
        if (info.video_url) {
          // show in panel
          videoPlayer.src = info.video_url;
          videoPlayer.style.display = "block";
          // post final video bubble with assistant text
          const html = `<div class=msg>
              <div>${escapeHtml(assistantText||'Your video is ready.')}</div>
              ${buildThinkingToggle(lastPlanResult)}
              <div class='chat-video'><video src='${info.video_url}' controls playsinline></video></div>
            </div>`;
          if (thinkingHandle) thinkingHandle.replace(html); else appendChat("assistant", html);
          setProcessing(false);
          break;
        }
        if (info.status && ["error","validation_error"].includes(String(info.status))) {
          if (thinkingHandle) thinkingHandle.error(info.message||String(info.status));
          setProcessing(false);
          break;
        }
        tries++;
        await new Promise(r => setTimeout(r, 3000));
      }
      if (tries >= 60) {
        if (thinkingHandle) thinkingHandle.error("Render did not produce a video in time.");
        setProcessing(false);
      }
    } catch(e) {
      if (thinkingHandle) thinkingHandle.error(String(e));
      setProcessing(false);
    }
  }

  function isImage(name){ return /\.(png|jpe?g|gif|webp|bmp)$/i.test(name||""); }
  function isVideo(name){ return /\.(mp4|webm|mov|mkv|avi)$/i.test(name||""); }
  function isAudio(name){ return /\.(mp3|wav|ogg|m4a|aac|flac)$/i.test(name||""); }

  function renderAttachments(items){
    if(!items || !items.length) return "";
    const html = items.map(it => {
      const n = it.name || it.path;
      if (it.url && isImage(n)) {
        return `<div class=\"att\"><img src=\"${it.url}\" alt=\"${escapeHtml(n)}\"/></div>`;
      }
      if (it.url && isVideo(n)) {
        return `<div class=\"att\"><video src=\"${it.url}\" muted controls preload=\"metadata\"></video></div>`;
      }
      if (it.url && isAudio(n)) {
        return `<div class=\"att\"><audio src=\"${it.url}\" controls preload=\"metadata\"></audio></div>`;
      }
      return `<div class=\"att\">${escapeHtml(n)}</div>`;
    }).join("");
    return `<div class=\"atts\">${html}</div>`;
  }

  function renderComposerAttachments(items){
    if(!items || !items.length) return "";
    const html = items.map(it => {
      const n = it.name || it.path;
      const media = it.url && isImage(n) ? `<img src=\"${it.url}\" alt=\"${escapeHtml(n)}\"/>`
                 : it.url && isVideo(n) ? `<video src=\"${it.url}\" muted preload=\"metadata\"></video>`
                 : it.url && isAudio(n) ? `<audio src=\"${it.url}\" preload=\"metadata\" controls></audio>`
                 : `<div class=\"name\">${escapeHtml(n)}</div>`;
      return `<div class=\"att\">${media}<button class=\"remove\" data-url=\"${it.url}\">✕</button></div>`;
    }).join("");
    return `<div class=\"atts\">${html}</div>`;
  }

  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }
})();
