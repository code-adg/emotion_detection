/* ═══════════════════════════════════════════════════════════════════════════
   EmotiSub — Redesigned Frontend Logic
   Intro animation · Custom cursor · Pipeline steps · Lazy loading
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── Splash Screen Animation ──────────────────────────────────────────────
  const splash = document.getElementById("splashScreen");
  const appWrapper = document.getElementById("appWrapper");

  setTimeout(() => {
    splash.classList.add("fade-out");
    appWrapper.classList.add("visible");
  }, 2400);

  setTimeout(() => {
    splash.style.display = "none";
  }, 3000);

  // ── Custom Cursor ────────────────────────────────────────────────────────
  const cursorDot = document.getElementById("cursorDot");
  const cursorRing = document.getElementById("cursorRing");
  let mouseX = 0, mouseY = 0;
  let ringX = 0, ringY = 0;

  document.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    cursorDot.style.left = mouseX + "px";
    cursorDot.style.top = mouseY + "px";
  });

  function animateRing() {
    ringX += (mouseX - ringX) * 0.15;
    ringY += (mouseY - ringY) * 0.15;
    cursorRing.style.left = ringX + "px";
    cursorRing.style.top = ringY + "px";
    requestAnimationFrame(animateRing);
  }
  animateRing();

  // Hover effect on interactive elements
  document.addEventListener("mouseover", (e) => {
    const target = e.target.closest("a, button, .upload-dropzone, .feature-card, .emotion-stat, input");
    if (target) document.body.classList.add("cursor-hover");
  });

  document.addEventListener("mouseout", (e) => {
    const target = e.target.closest("a, button, .upload-dropzone, .feature-card, .emotion-stat, input");
    if (target) document.body.classList.remove("cursor-hover");
  });

  // ── Lazy Loading (Intersection Observer) ─────────────────────────────────
  const lazyObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          lazyObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  document.querySelectorAll("[data-lazy]").forEach((el) => {
    lazyObserver.observe(el);
  });

  // ── DOM references ───────────────────────────────────────────────────────
  const uploadZone = document.getElementById("uploadZone");
  const srtInput = document.getElementById("srtInput");
  const fileNameEl = document.getElementById("fileName");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const pipelineSection = document.getElementById("pipelineSection");
  const resultsWrap = document.getElementById("resultsWrapper");
  const resultsBody = document.getElementById("resultsBody");
  const resultsCount = document.getElementById("resultsCount");
  const downloadBtn = document.getElementById("downloadBtn");
  const emotionSummary = document.getElementById("emotionSummary");

  let selectedFile = null;

  // ── Upload Zone Interactions ─────────────────────────────────────────────
  uploadZone.addEventListener("click", () => srtInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  srtInput.addEventListener("change", () => {
    if (srtInput.files.length > 0) handleFile(srtInput.files[0]);
  });

  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith(".srt")) {
      showError("Please select a valid .srt subtitle file.");
      return;
    }
    selectedFile = file;
    fileNameEl.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    analyzeBtn.disabled = false;

    // Hide old results
    resultsWrap.style.display = "none";
    downloadBtn.style.display = "none";
    pipelineSection.style.display = "none";
  }

  // ── Pipeline Step Animation ──────────────────────────────────────────────
  const pipelineSteps = [
    "step-parse",
    "step-tokenize",
    "step-predict",
    "step-threshold",
    "step-aggregate",
    "step-output",
  ];

  function resetPipeline() {
    pipelineSteps.forEach((id) => {
      const el = document.getElementById(id);
      el.classList.remove("active", "done");
    });
  }

  function activateStep(index) {
    if (index >= pipelineSteps.length) return;
    // Mark previous steps as done
    for (let i = 0; i < index; i++) {
      const el = document.getElementById(pipelineSteps[i]);
      el.classList.remove("active");
      el.classList.add("done");
    }
    // Mark current as active
    const curr = document.getElementById(pipelineSteps[index]);
    curr.classList.remove("done");
    curr.classList.add("active");
  }

  function completeAllSteps() {
    pipelineSteps.forEach((id) => {
      const el = document.getElementById(id);
      el.classList.remove("active");
      el.classList.add("done");
    });
  }

  // ── Analyze Button ──────────────────────────────────────────────────────
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show loading state
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    resultsWrap.style.display = "none";
    downloadBtn.style.display = "none";

    // Show and reset pipeline
    pipelineSection.style.display = "block";
    resetPipeline();

    // Animate pipeline steps progressively
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      activateStep(stepIndex);
      stepIndex++;
      if (stepIndex >= pipelineSteps.length) {
        clearInterval(stepInterval);
      }
    }, 800);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      clearInterval(stepInterval);
      completeAllSteps();

      if (!response.ok || data.error) {
        showError(data.error || "Server returned an error.");
        return;
      }

      // Short delay to show completed pipeline
      setTimeout(() => {
        renderResults(data);
      }, 500);
    } catch (err) {
      clearInterval(stepInterval);
      showError("Could not connect to the server. Is it running?");
      console.error(err);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  });

  // ── Render Results ──────────────────────────────────────────────────────
  function renderResults(data) {
    resultsBody.innerHTML = "";
    emotionSummary.innerHTML = "";
    resultsCount.textContent = `${data.count} subtitle${data.count !== 1 ? "s" : ""} analyzed`;

    // Build emotion summary
    const emotionCounts = {};
    const emotionColors = {
      anger: "var(--emo-anger)",
      disgust: "var(--emo-disgust)",
      fear: "var(--emo-fear)",
      joy: "var(--emo-joy)",
      sadness: "var(--emo-sadness)",
      surprise: "var(--emo-surprise)",
      neutral: "var(--emo-neutral)",
    };

    data.results.forEach((item) => {
      emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + 1;
    });

    // Sort by count desc
    const sortedEmotions = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]);

    sortedEmotions.forEach(([emotion, count]) => {
      const pct = ((count / data.count) * 100).toFixed(1);
      const color = emotionColors[emotion] || "var(--text-muted)";
      const card = document.createElement("div");
      card.className = "emotion-stat";
      card.innerHTML = `
        <span class="emotion-stat-label" style="color:${color}">${emotion}</span>
        <span class="emotion-stat-count" style="color:${color}">${count}</span>
        <span class="emotion-stat-pct">${pct}%</span>
      `;
      emotionSummary.appendChild(card);
    });

    // Render table rows with staggered animation
    data.results.forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.style.animationDelay = `${Math.min(index * 0.02, 1)}s`;

      const confPercent = (item.confidence * 100).toFixed(1);

      tr.innerHTML = `
        <td>${item.index}</td>
        <td style="white-space:nowrap; font-size:0.75rem; color:var(--text-muted);">${item.start} → ${item.end}</td>
        <td>${escapeHtml(item.text)}</td>
        <td><span class="emotion-pill emotion-pill--${item.emotion}">${item.emotion}</span></td>
        <td>
          <div class="conf-bar">
            <div class="conf-bar-track">
              <div class="conf-bar-fill" style="width:${confPercent}%"></div>
            </div>
            <span class="conf-value">${confPercent}%</span>
          </div>
        </td>
      `;

      resultsBody.appendChild(tr);
    });

    // Show results and download button
    resultsWrap.style.display = "block";
    if (data.download) {
      downloadBtn.href = data.download;
      downloadBtn.style.display = "inline-flex";
    }

    // Scroll to results
    setTimeout(() => {
      resultsWrap.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
  }

  // ── Helpers ─────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function showError(msg) {
    let toast = document.querySelector(".error-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "error-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 4000);
  }
})();
