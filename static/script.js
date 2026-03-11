/* ═══════════════════════════════════════════════════════════════════════════
   EmotiSub — Enhanced Frontend Logic  v2
   Splash · Cursor · Pipeline · Donut chart · Emoji pills · Counters ·
   Filter bar · Reset · Success toast · Keyboard a11y · Inline validation
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── Emoji map for 7 emotions ────────────────────────────────────────────
  const EMOTION_EMOJI = {
    anger: "😡",
    disgust: "🤢",
    fear: "😨",
    joy: "😊",
    sadness: "😢",
    surprise: "😲",
    neutral: "😐",
  };

  const EMOTION_COLORS_HEX = {
    anger: "#ef4444",
    disgust: "#a3735c",
    fear: "#a855f7",
    joy: "#eab308",
    sadness: "#3b82f6",
    surprise: "#f97316",
    neutral: "#71717a",
  };

  // ── Splash Screen Animation ──────────────────────────────────────────────
  const splash = document.getElementById("splashScreen");
  const appWrapper = document.getElementById("appWrapper");

  setTimeout(() => {
    splash.classList.add("fade-out");
    appWrapper.classList.add("visible");
  }, 2800);

  setTimeout(() => {
    splash.style.display = "none";
  }, 3400);

  // ── Scroll Indicator — hide on scroll ───────────────────────────────────
  const scrollIndicator = document.getElementById("scrollIndicator");

  window.addEventListener("scroll", () => {
    if (window.scrollY > 100) {
      scrollIndicator.style.opacity = "0";
      scrollIndicator.style.pointerEvents = "none";
    } else {
      scrollIndicator.style.opacity = "";
      scrollIndicator.style.pointerEvents = "";
    }
  }, { passive: true });

  // ── "Try It Now" — instant jump to analysis section ─────────────────────
  const heroCta = document.getElementById("heroCta");
  if (heroCta) {
    heroCta.addEventListener("click", (e) => {
      e.preventDefault();
      const target = document.getElementById("uploadAnchor");
      if (target) {
        // Disable smooth scroll temporarily for an instant jump
        document.documentElement.style.scrollBehavior = "auto";
        target.scrollIntoView({ behavior: "auto", block: "start" });
        // Re-enable smooth scroll after jump
        requestAnimationFrame(() => {
          document.documentElement.style.scrollBehavior = "";
        });
      }
    });
  }

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
    const target = e.target.closest(
      "a, button, .upload-dropzone, .feature-card, .emotion-stat, .filter-chip, input"
    );
    if (target) document.body.classList.add("cursor-hover");
  });

  document.addEventListener("mouseout", (e) => {
    const target = e.target.closest(
      "a, button, .upload-dropzone, .feature-card, .emotion-stat, .filter-chip, input"
    );
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
  const dropzoneError = document.getElementById("dropzoneError");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const pipelineSection = document.getElementById("pipelineSection");
  const resultsWrap = document.getElementById("resultsWrapper");
  const resultsBody = document.getElementById("resultsBody");
  const resultsCount = document.getElementById("resultsCount");
  const downloadBtn = document.getElementById("downloadBtn");
  const downloadEmojiBtn = document.getElementById("downloadEmojiBtn");
  const resetBtn = document.getElementById("resetBtn");
  const emotionSummary = document.getElementById("emotionSummary");
  const filterBar = document.getElementById("filterBar");
  const chartLegend = document.getElementById("chartLegend");
  const chartTotal = document.getElementById("chartTotal");
  const donutCanvas = document.getElementById("donutChart");

  let selectedFile = null;
  let currentResults = null; // Store results for filtering

  // ── Upload Zone Interactions ─────────────────────────────────────────────
  uploadZone.addEventListener("click", () => srtInput.click());

  // Keyboard accessibility
  uploadZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      srtInput.click();
    }
  });

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
    // Clear previous error
    dropzoneError.textContent = "";
    dropzoneError.innerHTML = "";

    if (!file.name.toLowerCase().endsWith(".srt")) {
      // Inline validation with shake
      dropzoneError.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i> Invalid file type. Please select a <strong>.srt</strong> subtitle file.`;
      uploadZone.classList.add("shake");
      setTimeout(() => uploadZone.classList.remove("shake"), 600);
      selectedFile = null;
      analyzeBtn.disabled = true;
      fileNameEl.textContent = "";
      return;
    }

    selectedFile = file;
    fileNameEl.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    analyzeBtn.disabled = false;

    // Hide old results
    resultsWrap.style.display = "none";
    downloadBtn.style.display = "none";
    resetBtn.style.display = "none";
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
    for (let i = 0; i < index; i++) {
      const el = document.getElementById(pipelineSteps[i]);
      el.classList.remove("active");
      el.classList.add("done");
    }
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

  // ── Reset / New Analysis ─────────────────────────────────────────────────
  resetBtn.addEventListener("click", () => {
    // Reset everything
    selectedFile = null;
    currentResults = null;
    srtInput.value = "";
    fileNameEl.textContent = "";
    dropzoneError.textContent = "";
    analyzeBtn.disabled = true;
    resultsWrap.style.display = "none";
    pipelineSection.style.display = "none";
    downloadBtn.style.display = "none";
    downloadEmojiBtn.style.display = "none";
    resetBtn.style.display = "none";

    // Scroll back to upload
    uploadZone.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // ── Analyze Button ──────────────────────────────────────────────────────
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Clear errors
    dropzoneError.textContent = "";

    // Show loading state
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    resultsWrap.style.display = "none";
    downloadBtn.style.display = "none";
    downloadEmojiBtn.style.display = "none";
    resetBtn.style.display = "none";

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
        showToast(data.error || "Server returned an error.", "error");
        return;
      }

      // Short delay to show completed pipeline
      setTimeout(() => {
        renderResults(data);
        showToast(`✅ ${data.count} subtitle${data.count !== 1 ? "s" : ""} analyzed successfully!`, "success");
      }, 500);
    } catch (err) {
      clearInterval(stepInterval);
      showToast("Could not connect to the server. Is it running?", "error");
      console.error(err);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  });

  // ── Render Results ──────────────────────────────────────────────────────
  function renderResults(data) {
    currentResults = data;
    resultsBody.innerHTML = "";
    emotionSummary.innerHTML = "";
    chartLegend.innerHTML = "";
    resultsCount.textContent = `${data.count} subtitle${data.count !== 1 ? "s" : ""} analyzed`;

    // Build emotion summary
    const emotionCounts = {};
    data.results.forEach((item) => {
      emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + 1;
    });

    // Sort by count desc
    const sortedEmotions = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]);

    // ── Render Emotion Summary Cards with animated counters & emoji ──
    sortedEmotions.forEach(([emotion, count]) => {
      const pct = ((count / data.count) * 100).toFixed(1);
      const color = EMOTION_COLORS_HEX[emotion] || "#71717a";
      const emoji = EMOTION_EMOJI[emotion] || "🔹";
      const card = document.createElement("div");
      card.className = "emotion-stat";
      card.innerHTML = `
        <span class="emotion-stat-emoji">${emoji}</span>
        <span class="emotion-stat-label" style="color:${color}">${emotion}</span>
        <span class="emotion-stat-count" style="color:${color}" data-target="${count}">0</span>
        <span class="emotion-stat-pct">${pct}%</span>
      `;
      emotionSummary.appendChild(card);
    });

    // Animate counters
    animateCounters();

    // ── Render Donut Chart ──
    drawDonutChart(sortedEmotions, data.count);

    // ── Render Chart Legend ──
    sortedEmotions.forEach(([emotion, count]) => {
      const color = EMOTION_COLORS_HEX[emotion] || "#71717a";
      const emoji = EMOTION_EMOJI[emotion] || "🔹";
      const item = document.createElement("div");
      item.className = "chart-legend-item";
      item.innerHTML = `
        <span class="chart-legend-dot" style="background:${color}"></span>
        <span class="chart-legend-label">${emoji} ${emotion}</span>
        <span class="chart-legend-value">${count}</span>
      `;
      chartLegend.appendChild(item);
    });

    // ── Conditionally show table for ≤ 30 lines, infographics-only for > 30 ──
    const tableContainer = document.querySelector(".table-container");
    const vlcNote = document.querySelector(".vlc-note");

    if (data.count <= 30) {
      // Show filter bar + table
      filterBar.style.display = "";
      tableContainer.style.display = "";
      if (vlcNote) vlcNote.style.display = "";

      buildFilterBar(sortedEmotions);
      renderTableRows(data.results);
    } else {
      // Hide filter bar + table — show only infographics
      filterBar.style.display = "none";
      tableContainer.style.display = "none";
      if (vlcNote) vlcNote.style.display = "none";
    }

    // Show results and buttons
    resultsWrap.style.display = "block";
    resetBtn.style.display = "inline-flex";
    if (data.download) {
      downloadBtn.href = data.download;
      downloadBtn.style.display = "inline-flex";
    }
    if (data.download_emoji) {
      downloadEmojiBtn.href = data.download_emoji;
      downloadEmojiBtn.style.display = "inline-flex";
    }

    // Re-observe for lazy loading
    const resultsCard = resultsWrap.querySelector(".results-card");
    if (resultsCard) lazyObserver.observe(resultsCard);

    // Scroll to results
    setTimeout(() => {
      resultsWrap.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
  }

  // ── Table Row Rendering ──────────────────────────────────────────────────
  function renderTableRows(results) {
    resultsBody.innerHTML = "";
    results.forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.style.animationDelay = `${Math.min(index * 0.02, 1)}s`;
      tr.dataset.emotion = item.emotion;

      const confPercent = (item.confidence * 100).toFixed(1);
      const emoji = EMOTION_EMOJI[item.emotion] || "";

      tr.innerHTML = `
        <td>${item.index}</td>
        <td style="white-space:nowrap; font-size:0.75rem; color:var(--text-muted);">${item.start} → ${item.end}</td>
        <td>${escapeHtml(item.text)}</td>
        <td><span class="emotion-pill emotion-pill--${item.emotion}"><span class="emoji">${emoji}</span> ${item.emotion}</span></td>
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
  }

  // ── Animated Number Counters ─────────────────────────────────────────────
  function animateCounters() {
    const counters = document.querySelectorAll(".emotion-stat-count[data-target]");
    counters.forEach((counter) => {
      const target = parseInt(counter.dataset.target, 10);
      const duration = 800; // ms
      const start = performance.now();

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        counter.textContent = Math.round(target * ease);
        if (progress < 1) requestAnimationFrame(tick);
        else counter.textContent = target;
      }
      requestAnimationFrame(tick);
    });
  }

  // ── Donut Chart (Canvas) ─────────────────────────────────────────────────
  function drawDonutChart(sortedEmotions, total) {
    chartTotal.textContent = total;
    const ctx = donutCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = 180;
    donutCanvas.width = size * dpr;
    donutCanvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const outerRadius = 80;
    const innerRadius = 55;

    let startAngle = -Math.PI / 2; // start from top

    // Animate the donut
    const animDuration = 1000;
    const animStart = performance.now();

    function drawFrame(now) {
      const elapsed = now - animStart;
      const progress = Math.min(elapsed / animDuration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);

      ctx.clearRect(0, 0, size, size);

      // Background ring
      ctx.beginPath();
      ctx.arc(cx, cy, outerRadius, 0, 2 * Math.PI);
      ctx.arc(cx, cy, innerRadius, 0, 2 * Math.PI, true);
      ctx.fillStyle = "rgba(255,255,255,0.03)";
      ctx.fill();

      let currentAngle = -Math.PI / 2;
      sortedEmotions.forEach(([emotion, count]) => {
        const sliceAngle = (count / total) * 2 * Math.PI * eased;
        const color = EMOTION_COLORS_HEX[emotion] || "#71717a";

        ctx.beginPath();
        ctx.arc(cx, cy, outerRadius, currentAngle, currentAngle + sliceAngle);
        ctx.arc(cx, cy, innerRadius, currentAngle + sliceAngle, currentAngle, true);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        currentAngle += sliceAngle;
      });

      if (progress < 1) requestAnimationFrame(drawFrame);
    }

    requestAnimationFrame(drawFrame);
  }

  // ── Filter Bar ───────────────────────────────────────────────────────────
  function buildFilterBar(sortedEmotions) {
    // Remove old dynamic chips (keep the "All" button)
    const allBtn = filterBar.querySelector('[data-emotion="all"]');
    filterBar.innerHTML = "";
    filterBar.appendChild(
      (() => {
        const label = document.createElement("span");
        label.className = "filter-bar-label";
        label.innerHTML = `<i class="bi bi-funnel"></i> Filter:`;
        return label;
      })()
    );
    const allChip = document.createElement("button");
    allChip.className = "filter-chip active";
    allChip.dataset.emotion = "all";
    allChip.textContent = "All";
    filterBar.appendChild(allChip);

    sortedEmotions.forEach(([emotion, count]) => {
      const chip = document.createElement("button");
      chip.className = "filter-chip";
      chip.dataset.emotion = emotion;
      const emoji = EMOTION_EMOJI[emotion] || "";
      chip.textContent = `${emoji} ${emotion} (${count})`;
      filterBar.appendChild(chip);
    });

    // Add click listeners
    filterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        filterBar.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        filterTable(chip.dataset.emotion);
      });
    });
  }

  function filterTable(emotion) {
    const rows = resultsBody.querySelectorAll("tr");
    rows.forEach((row) => {
      if (emotion === "all" || row.dataset.emotion === emotion) {
        row.classList.remove("hidden-row");
      } else {
        row.classList.add("hidden-row");
      }
    });
  }

  // ── Toast Notifications ──────────────────────────────────────────────────
  function showToast(msg, type = "error") {
    // Remove existing toasts
    document.querySelectorAll(".toast").forEach((t) => t.remove());

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = msg;
    document.body.appendChild(toast);

    // Trigger reflow for animation
    requestAnimationFrame(() => {
      toast.classList.add("visible");
    });

    setTimeout(() => {
      toast.classList.remove("visible");
      setTimeout(() => toast.remove(), 500);
    }, 4000);
  }

  // Also keep the legacy showError for compatibility
  function showError(msg) {
    showToast(msg, "error");
  }

  // ── Helpers ─────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }
  // ── Scroll Journey — scroll-synced animation ─────────────────────────────
  const sjContainer = document.getElementById("scrollJourney");
  const sjResearch = document.getElementById("sjResearch");
  const sjSteps = document.getElementById("sjSteps");
  const sjBrain = document.getElementById("sjBrain");
  const sjDot = document.getElementById("sjDot");
  const sjProgressFill = document.getElementById("sjProgressFill");
  const sjCards = document.querySelectorAll(".sj-card");

  // Number of step cards
  const CARD_COUNT = sjCards.length; // 5

  let sjTicking = false;

  window.addEventListener("scroll", () => {
    if (!sjTicking) {
      requestAnimationFrame(updateScrollJourney);
      sjTicking = true;
    }
  }, { passive: true });

  function updateScrollJourney() {
    sjTicking = false;

    if (!sjContainer) return;

    const rect = sjContainer.getBoundingClientRect();
    const containerHeight = sjContainer.offsetHeight;
    const viewportH = window.innerHeight;

    // Progress: 0 at top of container, 1 at bottom
    // The sticky element occupies 100vh, container is 500vh,
    // so effective scroll range = containerHeight - viewportH
    const scrollRange = containerHeight - viewportH;
    const scrolled = -rect.top; // How far we've scrolled into the container
    const progress = Math.max(0, Math.min(1, scrolled / scrollRange));

    /*
     * Phase timeline (mapped to progress 0–1):
     * 0.00 – 0.10: Research doc fades IN
     * 0.10 – 0.18: Research doc visible
     * 0.18 – 0.25: Research doc fades OUT
     * 0.25 – 0.30: Step cards fade IN
     * 0.30 – 0.75: "You" dot moves across cards (card progress)
     * 0.75 – 0.82: Step cards fade OUT
     * 0.82 – 0.90: Brain card fades IN
     * 0.90 – 1.00: Brain card visible
     */

    // ── Phase 1: Research ──
    if (progress < 0.25) {
      let researchOpacity = 0;
      if (progress < 0.10) {
        researchOpacity = progress / 0.10; // Fade in
      } else if (progress < 0.18) {
        researchOpacity = 1; // Hold
      } else {
        researchOpacity = 1 - ((progress - 0.18) / 0.07); // Fade out
      }
      sjResearch.style.opacity = Math.max(0, Math.min(1, researchOpacity));
      sjResearch.style.transform = `translateY(${(1 - researchOpacity) * 20}px)`;
      sjResearch.classList.toggle("active", researchOpacity > 0.01);
    } else {
      sjResearch.classList.remove("active");
      sjResearch.style.opacity = "0";
    }

    // ── Phase 2: Step Cards ──
    if (progress >= 0.22 && progress < 0.80) {
      let stepsOpacity = 1;
      if (progress < 0.30) {
        stepsOpacity = (progress - 0.22) / 0.08; // Fade in
      } else if (progress > 0.72) {
        stepsOpacity = 1 - ((progress - 0.72) / 0.08); // Fade out
      }
      stepsOpacity = Math.max(0, Math.min(1, stepsOpacity));
      sjSteps.style.opacity = stepsOpacity;
      sjSteps.style.transform = stepsOpacity < 1 && progress < 0.30
        ? `translateY(${(1 - stepsOpacity) * 20}px)`
        : `translateY(0)`;
      sjSteps.classList.toggle("active", stepsOpacity > 0.01);

      // Move the "You" dot across cards
      if (progress >= 0.30 && progress <= 0.72) {
        const dotProgress = (progress - 0.30) / 0.42; // 0 to 1

        // Map dot progress to track position (10% to 90% of track width)
        const trackStart = 10; // percent
        const trackEnd = 90;   // percent
        const dotLeft = trackStart + dotProgress * (trackEnd - trackStart);
        sjDot.style.left = dotLeft + "%";

        // Fill width matches dot position
        const fillWidth = dotProgress * (trackEnd - trackStart);
        sjProgressFill.style.width = fillWidth + "%";

        // Highlight the card the dot is currently over
        const cardIndex = Math.min(
          CARD_COUNT - 1,
          Math.floor(dotProgress * CARD_COUNT)
        );
        sjCards.forEach((card, i) => {
          card.classList.toggle("highlight", i === cardIndex);
        });
      }
    } else {
      sjSteps.classList.remove("active");
      sjSteps.style.opacity = "0";
      sjCards.forEach((c) => c.classList.remove("highlight"));
    }

    // ── Phase 3: Brain ──
    if (progress >= 0.78) {
      let brainOpacity = 0;
      if (progress < 0.88) {
        brainOpacity = (progress - 0.78) / 0.10; // Fade in
      } else {
        brainOpacity = 1; // Hold
      }
      brainOpacity = Math.max(0, Math.min(1, brainOpacity));
      sjBrain.style.opacity = brainOpacity;
      sjBrain.style.transform = `translateY(${(1 - brainOpacity) * 20}px)`;
      sjBrain.classList.toggle("active", brainOpacity > 0.01);
    } else {
      sjBrain.classList.remove("active");
      sjBrain.style.opacity = "0";
    }
  }

  // Initial call
  updateScrollJourney();
})();
