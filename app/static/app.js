const chatEl = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const btnClear = document.getElementById("btnClear");

function addMsg(role, text, meta = "") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  metaEl.textContent = meta || role;

  wrap.appendChild(bubble);
  wrap.appendChild(metaEl);

  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

function setTyping(show) {
  const existing = document.getElementById("typing");
  if (show) {
    if (existing) return;
    const wrap = addMsg("bot", "Sedang mengetik...", "bot");
    wrap.id = "typing";
    wrap.querySelector(".bubble").classList.add("typing");
  } else {
    if (existing) existing.remove();
  }
}

async function sendMessage(message) {
  setTyping(true);
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(`HTTP ${res.status} - ${t}`);
    }

    const data = await res.json();

    // tampilkan jawaban
    const source = data.source || "none";
    const conf = typeof data.confidence === "number" ? data.confidence.toFixed(2) : "";
    const meta = conf ? `${source} • conf ${conf}` : source;

    addMsg("bot", data.answer || "(kosong)", meta);

    // (opsional) tampilkan contexts ringkas di console
    if (Array.isArray(data.contexts) && data.contexts.length) {
      console.log("contexts:", data.contexts);
    }
  } catch (err) {
    addMsg("bot", `Terjadi error: ${err.message}`, "error");
  } finally {
    setTyping(false);
  }
}

btnClear.addEventListener("click", () => {
  // hapus semua chat kecuali pesan pertama
  const nodes = Array.from(chatEl.querySelectorAll(".msg"));
  nodes.slice(1).forEach(n => n.remove());
  input.focus();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMsg("user", text, "you");
  input.value = "";
  input.focus();

  await sendMessage(text);
});

// fokus input saat load
input.focus();
