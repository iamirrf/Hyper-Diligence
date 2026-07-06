const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const elements = {
  askForm: $("#ask-form"),
  questionInput: $("#question-input"),
  answerOutput: $("#answer-output"),
  citationList: $("#citation-list"),
  citationCount: $("#citation-count"),
  clearAnswer: $("#clear-answer"),
  searchForm: $("#search-form"),
  searchQuery: $("#search-query"),
  tickerFilter: $("#ticker-filter"),
  formFilter: $("#form-filter"),
  topK: $("#top-k"),
  kValue: $("#k-value"),
  resultCount: $("#result-count"),
  resultsList: $("#results-list"),
  healthText: $("#health-text"),
  statusPulse: $(".pulse"),
  chunkCount: $("#chunk-count"),
  evalsTable: $("#evals-table"),
  refreshEvals: $("#refresh-evals"),
  bestMode: $("#best-mode"),
  bestModeDetail: $("#best-mode-detail"),
};

const formatNumber = new Intl.NumberFormat("en-US");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setLoading(target, loading) {
  target.classList.toggle("loading", loading);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json();
}

function renderAnswer(text) {
  const lines = String(text || "").split("\n").filter(Boolean);
  if (!lines.length) {
    elements.answerOutput.innerHTML = '<p class="empty-state">No answer returned.</p>';
    return;
  }
  const bullets = lines.filter((line) => line.trim().startsWith("- "));
  const lead = lines.filter((line) => !line.trim().startsWith("- "));
  const leadHtml = lead.map((line) => `<p>${escapeHtml(line)}</p>`).join("");
  const bulletHtml = bullets.length
    ? `<ul>${bullets.map((line) => `<li>${escapeHtml(line.replace(/^- /, ""))}</li>`).join("")}</ul>`
    : "";
  elements.answerOutput.innerHTML = `<div class="answer-content">${leadHtml}${bulletHtml}</div>`;
}

function renderCitations(citations) {
  if (!citations.length) {
    elements.citationList.innerHTML = '<p class="empty-state compact">No citations returned.</p>';
    elements.citationCount.textContent = "0 citations";
    return;
  }
  elements.citationCount.textContent = `${citations.length} citation${citations.length === 1 ? "" : "s"}`;
  elements.citationList.innerHTML = citations
    .map(
      (item) => `
        <article class="citation-card">
          <div class="source-line">
            <span class="ticker-tag">${escapeHtml(item.ticker || "-")}</span>
            <strong>${escapeHtml(item.form || "-")}</strong>
            <span>${escapeHtml(item.filed || "-")}</span>
          </div>
          <div>${escapeHtml(item.section || "Unknown section")}</div>
          <span class="chunk-id">Chunk ${escapeHtml(item.chunk_id)}</span>
        </article>
      `,
    )
    .join("");
}

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    elements.questionInput.focus();
    return;
  }
  setLoading(elements.askForm, true);
  elements.answerOutput.innerHTML = '<p class="empty-state">Retrieving and composing a cited answer...</p>';
  elements.citationList.innerHTML = '<p class="empty-state compact">Collecting citations...</p>';
  try {
    const payload = await requestJson("/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    renderAnswer(payload.answer);
    renderCitations(payload.citations || []);
  } catch (error) {
    elements.answerOutput.innerHTML = `<p class="empty-state">Ask failed: ${escapeHtml(error.message)}</p>`;
    renderCitations([]);
  } finally {
    setLoading(elements.askForm, false);
  }
}

function selectedMode() {
  return $("input[name='mode']:checked").value;
}

function scoreWidth(score) {
  if (!Number.isFinite(score)) return 0;
  const normalized = score > 1 ? Math.min(score / 20, 1) : Math.max(0, Math.min(1, score));
  return Math.round(normalized * 100);
}

function renderResults(results) {
  elements.resultCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;
  if (!results.length) {
    elements.resultsList.innerHTML = '<p class="empty-state compact">No matching passages found.</p>';
    return;
  }
  elements.resultsList.innerHTML = results
    .map(
      (item, index) => `
        <article class="result-card">
          <div class="result-meta">
            <span class="ticker-tag">${escapeHtml(item.ticker)}</span>
            <strong>#${index + 1}</strong>
            <span>${escapeHtml(item.form)} filed ${escapeHtml(item.filed)}</span>
            <span>${escapeHtml(item.section || "Unknown section")}</span>
            <span>score ${Number(item.score).toFixed(3)}</span>
          </div>
          <p>${escapeHtml(item.content)}</p>
          <div class="score-meter" aria-hidden="true"><span style="width: ${scoreWidth(Number(item.score))}%"></span></div>
        </article>
      `,
    )
    .join("");
}

async function runSearch() {
  const query = elements.searchQuery.value.trim();
  if (!query) {
    elements.searchQuery.focus();
    return;
  }
  const params = new URLSearchParams({
    q: query,
    mode: selectedMode(),
    k: elements.topK.value,
  });
  if (elements.tickerFilter.value) params.set("ticker", elements.tickerFilter.value);
  if (elements.formFilter.value) params.set("form", elements.formFilter.value);

  setLoading(elements.searchForm, true);
  elements.resultsList.innerHTML = '<p class="empty-state compact">Ranking filing passages...</p>';
  try {
    const payload = await requestJson(`/search?${params.toString()}`);
    renderResults(payload);
  } catch (error) {
    elements.resultCount.textContent = "Search failed";
    elements.resultsList.innerHTML = `<p class="empty-state compact">Search failed: ${escapeHtml(error.message)}</p>`;
  } finally {
    setLoading(elements.searchForm, false);
  }
}

async function loadHealth() {
  try {
    const payload = await requestJson("/health");
    elements.statusPulse.classList.add("ok");
    elements.statusPulse.classList.remove("error");
    elements.healthText.textContent = `${formatNumber.format(payload.chunks)} chunks online`;
    elements.chunkCount.textContent = formatNumber.format(payload.chunks);
  } catch (error) {
    elements.statusPulse.classList.add("error");
    elements.statusPulse.classList.remove("ok");
    elements.healthText.textContent = "API unavailable";
    elements.chunkCount.textContent = "-";
  }
}

function renderEvals(payload) {
  const modes = payload.modes || {};
  const entries = Object.entries(modes);
  if (!entries.length) {
    elements.evalsTable.innerHTML = '<p class="empty-state compact">No eval data found.</p>';
    elements.bestMode.textContent = "-";
    elements.bestModeDetail.textContent = "waiting for evals";
    return;
  }
  const best = entries
    .slice()
    .sort((a, b) => (b[1].recall_at_5 || 0) - (a[1].recall_at_5 || 0) || (b[1].mrr || 0) - (a[1].mrr || 0))[0];
  elements.bestMode.textContent = best[0].replaceAll("_", " ");
  elements.bestModeDetail.textContent = `Recall@5 ${(best[1].recall_at_5 || 0).toFixed(3)}`;
  elements.evalsTable.innerHTML = `
    <div class="eval-grid">
      <div class="eval-row eval-head">
        <span>Mode</span><span>Recall@5</span><span>Recall@10</span><span>MRR</span>
      </div>
      ${entries
        .map(([mode, metrics]) => {
          const values = [metrics.recall_at_5 || 0, metrics.recall_at_10 || 0, metrics.mrr || 0];
          return `
            <div class="eval-row">
              <strong>${escapeHtml(mode.replaceAll("_", " "))}</strong>
              ${values
                .map(
                  (value) => `
                    <div class="eval-cell">
                      <span>${value.toFixed(3)}</span>
                      <div class="bar" aria-hidden="true"><span style="width: ${Math.round(value * 100)}%"></span></div>
                    </div>
                  `,
                )
                .join("")}
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

async function loadEvals() {
  elements.evalsTable.innerHTML = '<p class="empty-state compact">Loading evals...</p>';
  try {
    renderEvals(await requestJson("/evals"));
  } catch (error) {
    elements.evalsTable.innerHTML = `<p class="empty-state compact">Eval load failed: ${escapeHtml(error.message)}</p>`;
  }
}

function wireEvents() {
  elements.askForm.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion();
  });
  elements.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });
  elements.clearAnswer.addEventListener("click", () => {
    elements.answerOutput.innerHTML = '<p class="empty-state">Run a question to see a cited answer pulled from the ingested SEC filings.</p>';
    elements.citationList.innerHTML = '<p class="empty-state compact">Citations will appear here with ticker, form, date, and section.</p>';
    elements.citationCount.textContent = "0 citations";
  });
  elements.topK.addEventListener("input", () => {
    elements.kValue.textContent = elements.topK.value;
  });
  elements.refreshEvals.addEventListener("click", loadEvals);
  $$(".sample").forEach((button) => {
    button.addEventListener("click", () => {
      elements.questionInput.value = button.dataset.question;
      elements.questionInput.focus();
    });
  });
}

wireEvents();
loadHealth();
loadEvals();
runSearch();
