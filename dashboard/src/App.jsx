import { useState, useCallback, useEffect, useRef } from "react";

const API_BASE = "http://127.0.0.1:8000";
const BANKS = ["HDFC", "SBI", "ICICI", "Axis", "Kotak", "BOB", "PNB", "YesBank"];
const NETWORK_TYPES = ["4G", "3G", "2G", "wifi"];
const DEVICE_TYPES = ["android", "ios", "feature_phone"];

const RISK_META = {
  HIGH:   { label: "High Risk",   color: "#DC2626", light: "#FEF2F2" },
  MEDIUM: { label: "Medium Risk", color: "#D97706", light: "#FFFBEB" },
  LOW:    { label: "Low Risk",    color: "#16A34A", light: "#F0FDF4" },
};

// ── Donut Chart via Chart.js ──────────────────────────────────────
function DonutGauge({ prob }) {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);
  const pct   = Math.round((prob || 0) * 100);
  const color = prob >= 0.6 ? "#DC2626" : prob >= 0.35 ? "#D97706" : "#16A34A";

  useEffect(() => {
    if (!canvasRef.current || typeof Chart === "undefined") return;
    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current, {
      type: "doughnut",
      data: { datasets: [{ data: [pct, 100 - pct],
        backgroundColor: [color, "#F1F5F9"], borderWidth: 0, borderRadius: [4, 0] }] },
      options: { cutout: "78%", rotation: -90, circumference: 180,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 700, easing: "easeInOutQuart" } },
    });
    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [prob, color, pct]);

  return (
    <div style={{ position: "relative", width: 160, height: 90, margin: "0 auto" }}>
      <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0 }} />
      <div style={{ position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)",
        textAlign: "center", lineHeight: 1 }}>
        <div style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", fontSize: 28,
          fontWeight: 800, color, letterSpacing: "-1px" }}>{pct}%</div>
        <div style={{ fontSize: 10, color: "#94A3B8", fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.05em", marginTop: 2 }}>FAILURE PROB.</div>
      </div>
    </div>
  );
}

// ── Feature bar chart ─────────────────────────────────────────────
function FeatureChart({ factors }) {
  if (!factors?.length) return null;
  const max = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {factors.slice(0, 6).map((f, i) => {
        const w        = (Math.abs(f.shap_value) / max) * 100;
        const pos      = f.shap_value > 0;
        const barColor = pos ? "#DC2626" : "#16A34A";
        const label    = f.feature.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
        return (
          <div key={i}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 11, color: "#475569", fontWeight: 500,
                fontFamily: "'Plus Jakarta Sans', sans-serif" }}>{label}</span>
              <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                color: barColor, fontWeight: 500 }}>
                {pos ? "+" : ""}{f.shap_value.toFixed(3)}
              </span>
            </div>
            <div style={{ height: 6, background: "#F1F5F9", borderRadius: 99, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${w}%`, background: barColor,
                borderRadius: 99, transition: "width 0.6s ease" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Reusable components ───────────────────────────────────────────
function Input({ label, hint, children }) {
  return (
    <div>
      <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "#374151",
        fontFamily: "'Plus Jakarta Sans', sans-serif", marginBottom: 6 }}>{label}</label>
      {children}
      {hint && <p style={{ fontSize: 11, color: "#9CA3AF", marginTop: 4 }}>{hint}</p>}
    </div>
  );
}

const iStyle = {
  width: "100%", padding: "9px 12px", border: "1.5px solid #E5E7EB", borderRadius: 8,
  fontSize: 13, color: "#111827", outline: "none", fontFamily: "'Plus Jakarta Sans', sans-serif",
  background: "#fff", transition: "border-color 0.15s", boxSizing: "border-box",
};

function Badge({ text, color, bg }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 10px",
      borderRadius: 99, fontSize: 11, fontWeight: 600, background: bg, color,
      fontFamily: "'Plus Jakarta Sans', sans-serif", letterSpacing: "0.03em" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0 }} />
      {text}
    </span>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: "#fff", border: "1.5px solid #E5E7EB", borderRadius: 12, padding: "16px 20px" }}>
      <p style={{ fontSize: 11, color: "#6B7280", fontWeight: 600, textTransform: "uppercase",
        letterSpacing: "0.06em", marginBottom: 6 }}>{label}</p>
      <p style={{ fontSize: 24, fontWeight: 800, color: accent || "#111827",
        letterSpacing: "-0.5px", lineHeight: 1 }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: "#9CA3AF", marginTop: 4 }}>{sub}</p>}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────
export default function App() {
  const [form, setForm] = useState({
    sender_bank: "HDFC", receiver_bank: "SBI", amount: "",
    hour_of_day: new Date().getHours(), network_type: "4G",
    device_type: "android", is_salary_day: 0, is_festival_day: 0,
  });
  const [result,         setResult]         = useState(null);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState(null);
  const [history,        setHistory]        = useState([]);
  const [activeTab,      setActiveTab]      = useState("predict");
  const [time,           setTime]           = useState(new Date());
  const [bankHealth,     setBankHealth]     = useState({});
  const [bankMeta,       setBankMeta]       = useState({});
  const [healthFetchedAt,setHealthFetchedAt]= useState("");

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));
  const cfg  = result ? (RISK_META[result.risk_level] || RISK_META.MEDIUM) : null;
  const high = history.filter(h => h.risk_level === "HIGH").length;
  const med  = history.filter(h => h.risk_level === "MEDIUM").length;
  const low  = history.filter(h => h.risk_level === "LOW").length;

  // Clock + bank health polling
  useEffect(() => {
    const clock = setInterval(() => setTime(new Date()), 1000);

    const fetchHealth = async () => {
      try {
        const res  = await fetch(`${API_BASE}/bank-health`);
        const data = await res.json();
        const scores = {}, meta = {};
        Object.entries(data.banks).forEach(([bank, info]) => {
          scores[bank] = info.score;
          meta[bank]   = info;
        });
        setBankHealth(scores);
        setBankMeta(meta);
        setHealthFetchedAt(data.fetched_at);
      } catch (e) { console.error("Bank health fetch failed", e); }
    };

    fetchHealth();
    const healthTimer = setInterval(fetchHealth, 5 * 60 * 1000);
    return () => { clearInterval(clock); clearInterval(healthTimer); };
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!form.amount) { setError("Amount is required."); return; }
    setError(null); setLoading(true); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form, amount: parseFloat(form.amount),
          hour_of_day: parseInt(form.hour_of_day),
          is_salary_day: parseInt(form.is_salary_day),
          is_festival_day: parseInt(form.is_festival_day),
        }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail?.[0]?.msg || e.detail || "Prediction failed");
      }
      const data = await res.json();
      setResult(data);
      setHistory(h => [{ ...data, _amount: form.amount, _sender: form.sender_bank,
        _receiver: form.receiver_bank, _time: new Date().toLocaleTimeString() }, ...h].slice(0, 30));
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [form]);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        html, body, #root { height: 100%; }
        body { background: #F8FAFC; color: #111827; font-family: 'Plus Jakarta Sans', sans-serif; -webkit-font-smoothing: antialiased; }
        input:focus, select:focus { border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.08); }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 99px; }
        select { appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236B7280' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 36px !important; cursor: pointer; }
        input[type=range] { -webkit-appearance: none; height: 4px; background: #E5E7EB; border-radius: 99px; outline: none; cursor: pointer; width: 100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: #2563EB; border-radius: 50%; cursor: pointer; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideup { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        .btn-primary:hover { background: #1D4ED8 !important; }
        .nav-tab { cursor: pointer; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; transition: all 0.15s; border: none; }
        .nav-tab:hover { background: #F1F5F9; }
        .history-row:hover { background: #F8FAFC !important; }
      `}</style>

      {/* ── Navbar ── */}
      <nav style={{ background: "#fff", borderBottom: "1px solid #E5E7EB", padding: "0 32px",
        height: 60, display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, background: "#2563EB", borderRadius: 8,
            display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ color: "#fff", fontSize: 14, fontWeight: 800 }}>₹</span>
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#111827", lineHeight: 1.2 }}>
              UPI Failure Predictor
            </div>
            <div style={{ fontSize: 11, color: "#6B7280" }}>Intelligent Retry Engine</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{ display: "flex", gap: 4 }}>
            {["predict", "analytics"].map(tab => (
              <button key={tab} className="nav-tab" onClick={() => setActiveTab(tab)}
                style={{ background: activeTab === tab ? "#EFF6FF" : "transparent",
                  color: activeTab === tab ? "#2563EB" : "#6B7280" }}>
                {tab === "predict" ? "⚡ Predict" : "📊 Analytics"}
              </button>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "#6B7280", fontFamily: "'JetBrains Mono', monospace" }}>
            {time.toLocaleTimeString("en-IN", { hour12: false })}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 12px",
            background: "#F0FDF4", borderRadius: 99, border: "1px solid #BBF7D0" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#16A34A", display: "inline-block" }} />
            <span style={{ fontSize: 12, color: "#15803D", fontWeight: 600 }}>API Live</span>
          </div>
        </div>
      </nav>

      {activeTab === "predict" ? (
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 32px" }}>

          {history.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
              <StatCard label="Total Analysed" value={history.length} sub="this session" />
              <StatCard label="High Risk" value={high} sub={`${Math.round(high/history.length*100)}% of total`} accent="#DC2626" />
              <StatCard label="Medium Risk" value={med} sub={`${Math.round(med/history.length*100)}% of total`} accent="#D97706" />
              <StatCard label="Low Risk" value={low} sub={`${Math.round(low/history.length*100)}% of total`} accent="#16A34A" />
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24 }}>

            {/* Form */}
            <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB", overflow: "hidden" }}>
              <div style={{ padding: "20px 24px", borderBottom: "1px solid #F1F5F9" }}>
                <h2 style={{ fontSize: 16, fontWeight: 700, color: "#111827" }}>Transaction Details</h2>
                <p style={{ fontSize: 12, color: "#6B7280", marginTop: 2 }}>Enter parameters to predict failure risk</p>
              </div>
              <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
                <Input label="Transaction Amount (₹)" hint="Maximum ₹2,00,000 per UPI transaction">
                  <input style={iStyle} type="number" placeholder="e.g. 5,000"
                    value={form.amount} onChange={e => set("amount", e.target.value)} />
                </Input>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Input label="Sender Bank">
                    <select style={iStyle} value={form.sender_bank} onChange={e => set("sender_bank", e.target.value)}>
                      {BANKS.map(b => <option key={b}>{b}</option>)}
                    </select>
                  </Input>
                  <Input label="Receiver Bank">
                    <select style={iStyle} value={form.receiver_bank} onChange={e => set("receiver_bank", e.target.value)}>
                      {BANKS.map(b => <option key={b}>{b}</option>)}
                    </select>
                  </Input>
                </div>
                <div style={{ height: 1, background: "#F1F5F9" }} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Input label="Network Type">
                    <select style={iStyle} value={form.network_type} onChange={e => set("network_type", e.target.value)}>
                      {NETWORK_TYPES.map(n => <option key={n}>{n}</option>)}
                    </select>
                  </Input>
                  <Input label="Device Type">
                    <select style={iStyle} value={form.device_type} onChange={e => set("device_type", e.target.value)}>
                      {DEVICE_TYPES.map(d => <option key={d}>{d}</option>)}
                    </select>
                  </Input>
                </div>
                <Input label={`Hour of Day — ${form.hour_of_day}:00${((form.hour_of_day>=9&&form.hour_of_day<=11)||(form.hour_of_day>=19&&form.hour_of_day<=22))?" ⚠️ Peak":""}`}>
                  <input type="range" min={0} max={23} value={form.hour_of_day}
                    onChange={e => set("hour_of_day", parseInt(e.target.value))} />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4,
                    fontSize: 10, color: "#9CA3AF", fontFamily: "'JetBrains Mono', monospace" }}>
                    <span>12AM</span><span>6AM</span><span>12PM</span><span>6PM</span><span>11PM</span>
                  </div>
                </Input>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Input label="Salary Day?">
                    <select style={iStyle} value={form.is_salary_day} onChange={e => set("is_salary_day", parseInt(e.target.value))}>
                      <option value={0}>No</option><option value={1}>Yes (1st/2nd)</option>
                    </select>
                  </Input>
                  <Input label="Festival Day?">
                    <select style={iStyle} value={form.is_festival_day} onChange={e => set("is_festival_day", parseInt(e.target.value))}>
                      <option value={0}>No</option><option value={1}>Yes</option>
                    </select>
                  </Input>
                </div>
                {error && (
                  <div style={{ background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 8,
                    padding: "10px 14px", fontSize: 12, color: "#DC2626", display: "flex", gap: 8 }}>
                    <span>⚠️</span><span>{error}</span>
                  </div>
                )}
              </div>
              <div style={{ padding: "16px 24px", borderTop: "1px solid #F1F5F9" }}>
                <button className="btn-primary" onClick={handleSubmit} disabled={loading}
                  style={{ width: "100%", padding: "12px", background: loading ? "#93C5FD" : "#2563EB",
                    color: "#fff", border: "none", borderRadius: 10, cursor: loading ? "not-allowed" : "pointer",
                    fontSize: 14, fontWeight: 700, transition: "background 0.15s",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  {loading ? (
                    <><div style={{ width: 16, height: 16, border: "2px solid rgba(255,255,255,0.4)",
                      borderTop: "2px solid #fff", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                      Analysing transaction...</>
                  ) : "Predict Failure Risk →"}
                </button>
              </div>
            </div>

            {/* Result area */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {!result && !loading && (
                <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB",
                  padding: "60px 40px", display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", gap: 12 }}>
                  <div style={{ width: 56, height: 56, background: "#F1F5F9", borderRadius: 16,
                    display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>📊</div>
                  <h3 style={{ fontSize: 16, fontWeight: 700, color: "#374151" }}>No prediction yet</h3>
                  <p style={{ fontSize: 13, color: "#9CA3AF", textAlign: "center", maxWidth: 300 }}>
                    Fill in the transaction details and click Predict to see the risk analysis.
                  </p>
                </div>
              )}
              {loading && (
                <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB",
                  padding: "60px 40px", display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
                  <div style={{ width: 40, height: 40, border: "3px solid #EFF6FF",
                    borderTop: "3px solid #2563EB", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                  <div style={{ textAlign: "center" }}>
                    <p style={{ fontSize: 15, fontWeight: 600, color: "#374151" }}>Running inference</p>
                    <p style={{ fontSize: 12, color: "#9CA3AF", marginTop: 4 }}>XGBoost · SHAP · Retry Engine</p>
                  </div>
                </div>
              )}
              {result && cfg && (
                <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "slideup 0.3s ease" }}>
                  <div style={{ background: "#fff", borderRadius: 16, border: `1.5px solid ${cfg.color}33`,
                    padding: "20px 24px", borderLeft: `4px solid ${cfg.color}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
                          <Badge text={cfg.label} color={cfg.color} bg={cfg.light} />
                          {result.predicted_failure_code && (
                            <Badge text={`NPCI: ${result.predicted_failure_code}`} color="#4B5563" bg="#F9FAFB" />
                          )}
                          {result.cached && <Badge text="Cached" color="#2563EB" bg="#EFF6FF" />}
                        </div>
                        <h2 style={{ fontSize: 28, fontWeight: 800, color: cfg.color, letterSpacing: "-0.5px", lineHeight: 1 }}>
                          {Math.round(result.failure_probability * 100)}% Failure Probability
                        </h2>
                        <p style={{ fontSize: 13, color: "#6B7280", marginTop: 6 }}>
                          {form.sender_bank} → {form.receiver_bank} &nbsp;·&nbsp;
                          ₹{Number(form.amount).toLocaleString("en-IN")} &nbsp;·&nbsp;
                          {result.prediction_time_ms}ms
                        </p>
                      </div>
                      <DonutGauge prob={result.failure_probability} />
                    </div>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB", padding: "20px 24px" }}>
                      <h3 style={{ fontSize: 14, fontWeight: 700, color: "#111827", marginBottom: 4 }}>Feature Importance</h3>
                      <p style={{ fontSize: 11, color: "#9CA3AF", marginBottom: 16 }}>Top factors driving this prediction</p>
                      <FeatureChart factors={result.top_risk_factors} />
                    </div>
                    <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB",
                      padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
                      <div>
                        <h3 style={{ fontSize: 14, fontWeight: 700, color: "#111827", marginBottom: 4 }}>Retry Strategy</h3>
                        <p style={{ fontSize: 11, color: "#9CA3AF" }}>NPCI-based intelligent retry recommendation</p>
                      </div>
                      {result.retry_recommended ? (
                        <>
                          <div style={{ background: "#FFFBEB", border: "1px solid #FCD34D", borderRadius: 10, padding: "12px 14px" }}>
                            <p style={{ fontSize: 12, fontWeight: 600, color: "#92400E", marginBottom: 4 }}>Retry Recommended</p>
                            <p style={{ fontSize: 12, color: "#78350F", lineHeight: 1.5 }}>
                              {result.retry_suggestion || "Retry after a short delay."}
                            </p>
                          </div>
                          {result.retry_strategy && (
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                              <div style={{ background: "#F8FAFC", borderRadius: 8, padding: "10px 12px" }}>
                                <p style={{ fontSize: 10, color: "#9CA3AF", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>Max Attempts</p>
                                <p style={{ fontSize: 20, fontWeight: 800, color: "#111827", marginTop: 2 }}>{result.retry_strategy.max_attempts}</p>
                              </div>
                              <div style={{ background: "#F8FAFC", borderRadius: 8, padding: "10px 12px" }}>
                                <p style={{ fontSize: 10, color: "#9CA3AF", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>Wait (secs)</p>
                                <p style={{ fontSize: 20, fontWeight: 800, color: "#111827", marginTop: 2 }}>{result.retry_strategy.wait_seconds?.[0] || "—"}</p>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <div style={{ background: "#F0FDF4", border: "1px solid #86EFAC", borderRadius: 10, padding: "12px 14px" }}>
                          <p style={{ fontSize: 12, fontWeight: 600, color: "#15803D", marginBottom: 4 }}>✓ Transaction Looks Safe</p>
                          <p style={{ fontSize: 12, color: "#166534", lineHeight: 1.5 }}>Low failure risk. No retry strategy needed.</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <details style={{ background: "#fff", borderRadius: 12, border: "1.5px solid #E5E7EB", overflow: "hidden" }}>
                    <summary style={{ padding: "12px 20px", cursor: "pointer", fontSize: 12,
                      fontWeight: 600, color: "#6B7280", userSelect: "none" }}>
                      View Raw API Response
                    </summary>
                    <pre style={{ padding: "16px 20px", fontSize: 11, overflowX: "auto",
                      fontFamily: "'JetBrains Mono', monospace", color: "#374151",
                      lineHeight: 1.7, borderTop: "1px solid #F1F5F9", background: "#F8FAFC" }}>
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              )}

              {history.length > 0 && (
                <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB", overflow: "hidden" }}>
                  <div style={{ padding: "16px 20px", borderBottom: "1px solid #F1F5F9",
                    display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3 style={{ fontSize: 14, fontWeight: 700, color: "#111827" }}>Recent Transactions</h3>
                    <span style={{ fontSize: 12, color: "#6B7280" }}>{history.length} entries</span>
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "#F8FAFC" }}>
                        {["Route", "Amount", "Risk", "Probability", "Time"].map(h => (
                          <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontSize: 11,
                            fontWeight: 600, color: "#6B7280", textTransform: "uppercase",
                            letterSpacing: "0.06em", borderBottom: "1px solid #F1F5F9" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {history.slice(0, 8).map((item, i) => {
                        const r = RISK_META[item.risk_level] || RISK_META.MEDIUM;
                        return (
                          <tr key={i} className="history-row" style={{ borderBottom: "1px solid #F1F5F9",
                            cursor: "pointer", transition: "background 0.1s" }} onClick={() => setResult(item)}>
                            <td style={{ padding: "12px 16px", fontSize: 13, color: "#374151", fontWeight: 500 }}>
                              {item._sender} → {item._receiver}
                            </td>
                            <td style={{ padding: "12px 16px", fontSize: 13,
                              fontFamily: "'JetBrains Mono', monospace", color: "#111827" }}>
                              ₹{Number(item._amount).toLocaleString("en-IN")}
                            </td>
                            <td style={{ padding: "12px 16px" }}>
                              <Badge text={r.label} color={r.color} bg={r.light} />
                            </td>
                            <td style={{ padding: "12px 16px", fontSize: 13, fontWeight: 700,
                              color: r.color, fontFamily: "'JetBrains Mono', monospace" }}>
                              {Math.round(item.failure_probability * 100)}%
                            </td>
                            <td style={{ padding: "12px 16px", fontSize: 12, color: "#9CA3AF" }}>
                              {item._time}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>

      ) : (
        /* Analytics Tab */
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 32px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 24 }}>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 800, color: "#111827", marginBottom: 4 }}>Bank Health Monitor</h2>
              <p style={{ fontSize: 13, color: "#6B7280" }}>
                Live reliability scores — updated every 5 minutes based on peak hour and salary day stress
              </p>
            </div>
            {healthFetchedAt && (
              <span style={{ fontSize: 11, color: "#9CA3AF", fontFamily: "'JetBrains Mono', monospace" }}>
                Last updated: {healthFetchedAt}
              </span>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 28 }}>
            {Object.keys(bankHealth).length === 0 ? (
              <div style={{ gridColumn: "span 4", textAlign: "center", padding: 40, color: "#9CA3AF", fontSize: 13 }}>
                Loading bank health data...
              </div>
            ) : Object.entries(bankHealth).map(([bank, score]) => {
              const info  = bankMeta[bank] || {};
              const color = info.status === "Excellent" ? "#16A34A"
                : info.status === "Good" ? "#D97706" : "#DC2626";
              const label = info.status || "Unknown";
              return (
                <div key={bank} style={{ background: "#fff", borderRadius: 16,
                  border: "1.5px solid #E5E7EB", padding: "20px 24px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div style={{ fontSize: 15, fontWeight: 700, color: "#111827" }}>{bank}</div>
                    <Badge text={label} color={color} bg={color + "15"} />
                  </div>
                  <div style={{ fontSize: 36, fontWeight: 800, color, letterSpacing: "-1px", lineHeight: 1 }}>
                    {score}<span style={{ fontSize: 16, fontWeight: 500, color: "#9CA3AF" }}>/100</span>
                  </div>
                  <div style={{ height: 6, background: "#F1F5F9", borderRadius: 99, marginTop: 12, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${score}%`, background: color, borderRadius: 99 }} />
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 8, fontSize: 10,
                    fontFamily: "'JetBrains Mono', monospace", color: "#9CA3AF" }}>
                    {info.is_peak_hour && <span style={{ color: "#D97706" }}>⚠ Peak Hour</span>}
                    {info.is_salary_day && <span style={{ color: "#DC2626" }}>⚠ Salary Day</span>}
                    {!info.is_peak_hour && !info.is_salary_day && <span>Normal load</span>}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ background: "#fff", borderRadius: 16, border: "1.5px solid #E5E7EB", padding: "24px 28px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#111827", marginBottom: 4 }}>NPCI Failure Code Reference</h3>
            <p style={{ fontSize: 12, color: "#6B7280", marginBottom: 20 }}>Common failure codes and their retry strategies</p>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#F8FAFC" }}>
                  {["Code", "Description", "Cause", "Retry?"].map(h => (
                    <th key={h} style={{ padding: "10px 16px", textAlign: "left", fontSize: 11,
                      fontWeight: 600, color: "#6B7280", textTransform: "uppercase",
                      letterSpacing: "0.06em", borderBottom: "1px solid #F1F5F9" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ["U30", "Request Timeout",         "High network latency or bank slowness", true],
                  ["U09", "Remitter Bank Timeout",   "Sender bank under heavy load",          true],
                  ["BT",  "Bank Server Busy",        "Peak traffic or salary day",            true],
                  ["U16", "Risk Threshold Exceeded", "High amount or unusual pattern",        false],
                  ["U68", "Transaction Not Permitted","Bank-level restriction",               false],
                  ["Z9",  "Insufficient Funds",      "Low account balance",                   false],
                ].map(([code, desc, cause, retry]) => (
                  <tr key={code} style={{ borderBottom: "1px solid #F1F5F9" }}>
                    <td style={{ padding: "12px 16px" }}>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 600,
                        background: "#F1F5F9", padding: "3px 8px", borderRadius: 6, color: "#374151" }}>{code}</span>
                    </td>
                    <td style={{ padding: "12px 16px", fontSize: 13, fontWeight: 500, color: "#374151" }}>{desc}</td>
                    <td style={{ padding: "12px 16px", fontSize: 12, color: "#6B7280" }}>{cause}</td>
                    <td style={{ padding: "12px 16px" }}>
                      <Badge text={retry ? "Yes" : "No"}
                        color={retry ? "#16A34A" : "#DC2626"}
                        bg={retry ? "#F0FDF4" : "#FEF2F2"} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}