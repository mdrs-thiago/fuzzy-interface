const { useState, useEffect, useMemo, useRef } = React;
const { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ReferenceLine } = Recharts;

// --- Icons ---
const Icons = {
    Plus: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>,
    Trash: () => <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
    Play: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    Settings: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
    Download: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>,
    Upload: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>,
    X: () => <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>,
    Check: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>,
    Power: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
    Caret: () => <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
};

// --- Styled Components ---

const Button = ({ onClick, children, variant = "primary", size = "md", className = "", icon: Icon, disabled }) => {
    const base = "inline-flex items-center justify-center rounded-md font-semibold tracking-tight transition-all duration-200 focus:outline-none whitespace-nowrap active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none disabled:active:scale-100";

    const variants = {
        primary: "bg-puc-red text-white shadow-md hover:bg-[#a3442a] hover:-translate-y-[1px]",
        secondary: "bg-white text-puc-gray border border-zinc-200 shadow-sm hover:bg-zinc-50 hover:text-puc-red hover:border-puc-red/30 hover:-translate-y-[0.5px]",
        ghost: "text-zinc-500 hover:text-puc-red hover:bg-puc-red/5 transition-colors",
        danger: "bg-rose-50 text-rose-600 border border-rose-200 hover:bg-rose-100 hover:text-rose-700 hover:border-rose-300 hover:-translate-y-[0.5px]",
        action: "bg-zinc-900 text-white shadow-md hover:bg-zinc-800 hover:-translate-y-[1px]"
    };

    const sizes = {
        sm: "px-2.5 py-1.5 text-[11px]",
        md: "px-4 py-2 text-xs",
        lg: "px-6 py-3 text-sm",
        icon: "p-2 aspect-square"
    };

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
        >
            {Icon && <span className={`${children ? 'mr-2' : ''} opacity-90`}><Icon /></span>}
            {children}
        </button>
    );
};

const Input = ({ label, className = "", ...props }) => (
    <div className={`w-full ${className}`}>
        {label && <label className="block text-[10px] font-bold text-zinc-500 mb-1 uppercase tracking-wider">{label}</label>}
        <input {...props} className="w-full bg-white border border-zinc-200 rounded px-2 py-1.5 text-zinc-800 text-xs focus:border-puc-red focus:outline-none transition-colors placeholder-zinc-400 shadow-sm" />
    </div>
);

const Select = ({ label, children, ...props }) => (
    <div className="w-full">
        {label && <label className="block text-[10px] font-bold text-zinc-500 mb-1 uppercase tracking-wider">{label}</label>}
        <div className="relative">
            <select {...props} className="w-full appearance-none bg-white border border-zinc-200 rounded px-2 py-1.5 text-zinc-800 text-xs focus:border-puc-red focus:outline-none transition-colors cursor-pointer shadow-sm">
                {children}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none text-zinc-400"><Icons.Settings /></div>
        </div>
    </div>
);

// --- Config ---
const COLORS = ["#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#60a5fa"];

const downloadJSON = (data, filename) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
};

// ----------------------------------------------------------------------
// Main Application
// ----------------------------------------------------------------------

// --- Data-Driven View (Wang-Mendel) ---
function DataDrivenView({ config, refresh }) {
    const [csvData, setCsvData] = useState("");
    const [columns, setColumns] = useState([]);
    const [targetCol, setTargetCol] = useState("");
    const [nTerms, setNTerms] = useState(5);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const text = ev.target.result;
            setCsvData(text);
            const firstLine = text.split('\n')[0];
            if (firstLine) {
                const cols = firstLine.split(',').map(c => c.trim()).filter(c => c);
                setColumns(cols);
                if (cols.length > 0) setTargetCol(cols[cols.length - 1]);
            }
        };
        reader.readAsText(file);
    };

    const handleRun = async () => {
        if (!csvData) return alert("Please upload a CSV file first.");
        setLoading(true);
        try {
            // Encode as base64 to avoid JSON string escaping issues
            const utf8Bytes = new TextEncoder().encode(csvData);
            const b64 = btoa(String.fromCharCode(...utf8Bytes));
            const res = await fetch("/wang-mendel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ csv_data: b64, target_col: targetCol, n_terms: parseInt(nTerms) })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail);
            }
            const data = await res.json();
            setResult(data);
        } catch (e) {
            alert("Error: " + e.message);
        }
        setLoading(false);
    };

    const handleImport = async () => {
        if (!result) return;
        try {
            await fetch("/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(result) });
            refresh();
            alert("Rules imported to FIS Builder successfully! Check the Editor and Rules tabs.");
        } catch(e) { alert("Import failed: " + e.message); }
    };

    return (
        <div className="h-full w-full flex flex-col p-6 overflow-hidden">
            <div className="flex justify-between items-center mb-6 shrink-0">
                <div>
                    <h2 className="text-xl font-black text-zinc-800 flex items-center gap-2">
                        Wang-Mendel Rule Extraction
                    </h2>
                    <p className="text-sm text-zinc-500 font-medium">Extract fuzzy rules from numerical input-output data pairs.</p>
                </div>
            </div>
            
            <div className="flex flex-1 min-h-0 gap-6">
                {/* Left Panel: Configuration */}
                <div className="w-1/3 flex flex-col gap-4 overflow-y-auto">
                    <div className="bg-white p-5 rounded-2xl border border-zinc-200 shadow-sm flex flex-col flex-1 min-h-[150px] justify-center items-center relative overflow-hidden group hover:border-puc-red/50 transition-colors">
                        <input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                        <div className="flex flex-col items-center gap-3 pointer-events-none">
                            <div className="w-12 h-12 rounded-full bg-puc-red/10 flex items-center justify-center text-puc-red group-hover:scale-110 transition-transform">
                                <Icons.Upload />
                            </div>
                            <div className="text-center">
                                <h3 className="text-sm font-black text-zinc-800">Upload CSV File</h3>
                                <p className="text-xs text-zinc-500 mt-1">{columns.length > 0 ? `${columns.length} columns loaded` : "Drag and drop or click to browse"}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div className="bg-white p-5 rounded-2xl border border-zinc-200 shadow-sm">
                        <div className="flex flex-col gap-4">
                            <div>
                                <label className="block text-xs font-black text-zinc-500 uppercase mb-1">Target Column (Consequent)</label>
                                <select value={targetCol} onChange={e => setTargetCol(e.target.value)} disabled={columns.length === 0}
                                    className="w-full bg-zinc-50 border border-zinc-200 rounded-lg p-2 text-sm focus:outline-none focus:ring-2 focus:ring-puc-red/20 disabled:opacity-50">
                                    {columns.length === 0 ? <option>Upload data first...</option> : columns.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-xs font-black text-zinc-500 uppercase mb-1">Fuzzy Sets per Variable ({nTerms})</label>
                                <input type="range" min="3" max="9" step="2" value={nTerms} onChange={e => setNTerms(e.target.value)}
                                    className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red" />
                            </div>
                            <button 
                                onClick={handleRun}
                                disabled={loading}
                                className="w-full py-2.5 bg-puc-red hover:bg-[#8C3A22] text-white rounded-lg font-bold transition-colors disabled:opacity-50 mt-2">
                                {loading ? "Extracting..." : "Run Wang-Mendel"}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Panel: Results */}
                <div className="flex-1 bg-white p-5 rounded-2xl border border-zinc-200 shadow-sm flex flex-col min-h-0">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xs font-black text-puc-gray uppercase tracking-widest">Extracted Rule Base</h3>
                        {result && (
                            <button 
                                onClick={handleImport}
                                className="px-4 py-1.5 bg-zinc-800 hover:bg-black text-white rounded-md text-xs font-bold transition-colors">
                                Import to Studio
                            </button>
                        )}
                    </div>
                    <div className="flex-1 bg-zinc-50 rounded-lg border border-zinc-200 p-4 overflow-y-auto">
                        {!result ? (
                            <div className="h-full flex items-center justify-center text-zinc-400 text-sm font-medium">
                                Run extraction to view rules here.
                            </div>
                        ) : (
                            <div className="flex flex-col gap-3">
                                {result.rules.map((r, i) => {
                                    const ant = Object.entries(r.antecedents).map(([k, v]) => `${k} IS ${v}`).join(" AND ");
                                    const cons = Object.entries(r.consequent).map(([k, v]) => `${k} IS ${v}`).join(" AND ");
                                    return (
                                        <div key={i} className="flex flex-col gap-1 bg-white p-3 rounded-md border border-zinc-200 shadow-sm">
                                            <div className="flex justify-between items-center">
                                                <span className="text-[10px] font-black bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded">R{i}</span>
                                            </div>
                                            <div className="font-mono text-sm">
                                                <span className="text-blue-600">IF</span> {ant} <span className="text-puc-red">THEN</span> {cons}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

function App() {
    const [config, setConfig] = useState({ variables: {}, rules: [] });
    const [view, setView] = useState("editor"); // editor | rules | simulate
    const [selectedVarName, setSelectedVarName] = useState(null);

    const refresh = async () => {
        try {
            const res = await fetch("/config");
            const data = await res.json();
            setConfig(data);
            // Ensure valid selection
            if (view === 'editor' && !selectedVarName && Object.keys(data.variables).length > 0) {
                setSelectedVarName(Object.keys(data.variables)[0]);
            }
        } catch (e) { console.error(e); }
    };

    useEffect(() => { refresh(); }, [view]);

    // Import/Export
    const handleExport = async () => {
        const res = await fetch("/export"); downloadJSON(await res.json(), `fuzzy_system.json`);
    };
    const handleImport = (e) => {
        const file = e.target.files[0]; if (!file) return;
        const reader = new FileReader();
        reader.onload = async (ev) => {
            try { await fetch("/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: ev.target.result }); refresh(); } catch (e) { alert("Import failed"); }
        };
        reader.readAsText(file);
    };

    return (
        <div className="flex flex-col h-screen bg-[#F4F5F0] text-zinc-800 font-sans overflow-hidden">
            {/* Header / Toolbar */}
            <header className="h-12 border-b border-zinc-200 bg-white flex items-center justify-between px-4 shrink-0 z-10 shadow-sm">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-6 bg-puc-red rounded-sm flex items-center justify-center text-white text-[10px] font-black italic">PUC</div>
                        <span className="font-bold text-puc-red text-sm tracking-tight uppercase">Fuzzy Studio</span>
                    </div>
                    <nav className="flex gap-1">
                        {["Editor", "Rules", "Data-Driven", "Simulations"].map(v => (
                            <button key={v} onClick={() => setView(v.toLowerCase().replace(" ", "-"))}
                                className={`px-4 py-1.5 rounded-full text-xs font-semibold tracking-tight transition-all ${view === v.toLowerCase().replace(" ", "-") ? "bg-puc-red text-white shadow-sm" : "text-zinc-500 hover:text-puc-red hover:bg-puc-red/5"
                                    }`}>
                                {v}
                            </button>
                        ))}
                    </nav>
                </div>
                <div className="flex items-center gap-3">
                    <button onClick={handleExport} className="p-1.5 text-zinc-400 hover:text-puc-red transition-colors" title="Export"><Icons.Download /></button>
                    <label className="p-1.5 text-zinc-400 hover:text-puc-red transition-colors cursor-pointer" title="Import">
                        <Icons.Upload />
                        <input type="file" className="hidden" accept=".json" onChange={handleImport} />
                    </label>
                </div>
            </header>

            {/* Main Workspace */}
            <div className="flex-1 flex min-h-0">
                {view === "editor" && <IDEView config={config} refresh={refresh} selectedVarName={selectedVarName} setSelectedVarName={setSelectedVarName} />}
                {view === "rules" && <RulesView config={config} refresh={refresh} />}
                {view === "data-driven" && <DataDrivenView config={config} refresh={refresh} />}
                {view === "simulations" && <SimulateView config={config} refresh={refresh} />}
            </div>
        </div>
    );
}

// ----------------------------------------------------------------------
// View: IDE (Variables Sidebar + Editor Main)
// ----------------------------------------------------------------------
function IDEView({ config, refresh, selectedVarName, setSelectedVarName }) {
    const [creationState, setCreationState] = useState(null); // { role: 'input' | 'output' }
    const [createForm, setCreateForm] = useState({ name: "", min: 0, max: 10 });

    // Group variables
    const inputs = Object.keys(config.variables).filter(v => config.variables[v].role === 'input');
    const outputs = Object.keys(config.variables).filter(v => config.variables[v].role === 'output');

    const startCreation = (role) => {
        setCreationState({ role });
        setCreateForm({ name: "", min: 0, max: 10 });
    };

    const confirmCreation = async () => {
        if (!createForm.name) return;
        await fetch("/variable", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: createForm.name,
                min_val: parseFloat(createForm.min),
                max_val: parseFloat(createForm.max),
                role: creationState.role
            })
        });
        refresh();
        setSelectedVarName(createForm.name);
        setCreationState(null);
    };

    return (
        <>
            {/* Sidebar */}
            <aside className="w-64 border-r border-zinc-200 bg-white flex flex-col shrink-0 shadow-[1px_0_4px_rgba(0,0,0,0.02)]">
                <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-8">
                    {/* Inputs Group */}
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-[10px] font-black text-puc-gray uppercase tracking-[0.1em]">Input Variables</span>
                            <span className="text-[10px] font-bold text-puc-red bg-puc-red/10 px-2 py-0.5 rounded-full">{inputs.length}</span>
                        </div>
                        <div className="space-y-1 mb-3">
                            {inputs.map(v => (
                                <SidebarItem key={v} name={v} role="input" active={selectedVarName === v} onClick={() => setSelectedVarName(v)} />
                            ))}
                        </div>

                        {creationState?.role === 'input' ? (
                            <CreationForm form={createForm} setForm={setCreateForm} onConfirm={confirmCreation} onCancel={() => setCreationState(null)} />
                        ) : (
                            <button onClick={() => startCreation('input')} className="group w-full py-2.5 rounded-lg border border-dashed border-zinc-300 bg-zinc-50/50 text-[11px] font-bold text-zinc-400 hover:text-puc-red hover:bg-puc-red/[0.02] hover:border-puc-red/30 transition-all flex items-center justify-center gap-2">
                                <Icons.Plus />
                                <span>New Input</span>
                            </button>
                        )}
                    </div>

                    {/* Outputs Group */}
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-[10px] font-black text-puc-gray uppercase tracking-[0.1em]">Output Variables</span>
                            <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">{outputs.length}</span>
                        </div>
                        <div className="space-y-1 mb-3">
                            {outputs.map(v => (
                                <SidebarItem key={v} name={v} role="output" active={selectedVarName === v} onClick={() => setSelectedVarName(v)} />
                            ))}
                        </div>

                        {creationState?.role === 'output' ? (
                            <CreationForm form={createForm} setForm={setCreateForm} onConfirm={confirmCreation} onCancel={() => setCreationState(null)} />
                        ) : (
                            <button onClick={() => startCreation('output')} className="group w-full py-2.5 rounded-lg border border-dashed border-zinc-300 bg-zinc-50/50 text-[11px] font-bold text-zinc-400 hover:text-emerald-500 hover:bg-emerald-50 hover:border-emerald-200 transition-all flex items-center justify-center gap-2">
                                <Icons.Plus />
                                <span>New Output</span>
                            </button>
                        )}
                    </div>
                </div>
            </aside>

            {/* Main Area */}
            <main className="flex-1 min-w-0 bg-[#F4F5F0]/50">
                {selectedVarName && config.variables[selectedVarName] ? (
                    <VariableEditor name={selectedVarName} data={config.variables[selectedVarName]} refresh={refresh} />
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-700">
                        <div className="text-4xl opacity-20 mb-2"><Icons.Settings /></div>
                        <p>Select or create a variable to edit</p>
                    </div>
                )}
            </main>
        </>
    );
}

const SidebarItem = ({ name, role, active, onClick }) => (
    <div
        onClick={onClick}
        className={`px-3 py-2.5 rounded-lg cursor-pointer flex items-center gap-3 text-xs transition-all border ${active ? "bg-white text-puc-red border-puc-red/20 shadow-sm font-bold" : "text-puc-gray border-transparent hover:text-zinc-900 hover:bg-zinc-50"
            }`}
    >
        <div className={`w-1.5 h-1.5 rounded-full ${role === 'input' ? 'bg-puc-red' : 'bg-emerald-500'}`}></div>
        <span className="truncate flex-1">{name}</span>
        {active && <div className="w-1 h-3 bg-puc-red rounded-full"></div>}
    </div>
);

const CreationForm = ({ form, setForm, onConfirm, onCancel }) => (
    <div className="p-3 border border-zinc-200 bg-white rounded-lg space-y-3 animate-in fade-in zoom-in-95 duration-200 shadow-lg">
        <Input autoFocus placeholder="Variable Name" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
        <div className="grid grid-cols-2 gap-2">
            <Input type="number" label="Universe Min" value={form.min} onChange={e => setForm({ ...form, min: e.target.value })} />
            <Input type="number" label="Universe Max" value={form.max} onChange={e => setForm({ ...form, max: e.target.value })} />
        </div>
        <div className="flex justify-end gap-2 pt-1">
            <button onClick={onCancel} className="text-[11px] font-bold text-zinc-400 hover:text-puc-gray transition-colors px-2 py-1">Discard</button>
            <button onClick={onConfirm} className="text-[11px] font-bold bg-puc-red hover:bg-[#a3442a] text-white px-3 py-1.5 rounded shadow-sm transition-colors">Create Variable</button>
        </div>
    </div>
);

// --- Variable Editor Component ---
function VariableEditor({ name, data, refresh }) {
    const [plotData, setPlotData] = useState([]);
    const [termForm, setTermForm] = useState({ name: "", type: "tri", params: "" });
    const [editingTerm, setEditingTerm] = useState(null); // name of term being edited
    const [dragging, setDragging] = useState(null); // { term, type, index, originalParams }
    const chartRef = useRef(null);

    const fetchPlotData = () => {
        fetch(`/plot/${name}`).then(r => r.json()).then(d => {
            const fmt = d.x.map((x, i) => {
                const row = { x };
                Object.keys(d.terms).forEach(t => row[t] = d.terms[t][i]);
                return row;
            });
            setPlotData(fmt);
        });
    };

    useEffect(() => {
        fetchPlotData();
        const mid = (data.max + data.min) / 2;
        setTermForm(p => ({ ...p, params: `${data.min.toFixed(1)}, ${mid.toFixed(1)}, ${data.max.toFixed(1)}` }));
    }, [name, data]);

    const addTerm = async () => {
        const p = termForm.params.split(',').map(x => parseFloat(x));
        await fetch(`/variable/${name}/term`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: termForm.name || "Term", type: termForm.type, params: p })
        });
        setEditingTerm(null);
        setTermForm({ name: "", type: "tri", params: `${data.min.toFixed(1)}, ${((data.min + data.max) / 2).toFixed(1)}, ${data.max.toFixed(1)}` });
        refresh();
    };

    const startEditTerm = (tName) => {
        const term = data.terms[tName];
        if (!term) return;
        const typeMap = { 'TriangularMF': 'tri', 'TrapezoidalMF': 'trap', 'GaussianMF': 'gauss', 'BellMF': 'bell', 'SigmoidMF': 'sig' };
        const shortType = typeMap[term.type] || term.type.toLowerCase().replace('mf', '');
        setTermForm({ name: tName, type: shortType, params: term.params.join(', ') });
        setEditingTerm(tName);
    };

    const cancelEdit = () => {
        setEditingTerm(null);
        setTermForm({ name: "", type: "tri", params: `${data.min.toFixed(1)}, ${((data.min + data.max) / 2).toFixed(1)}, ${data.max.toFixed(1)}` });
    };

    const deleteTerm = async (t) => {
        await fetch(`/variable/${name}/term/${t}`, { method: "DELETE" });
        refresh();
    };

    // --- MF Math for Live Preview ---
    const triMF = (x, a, b, c) => Math.max(0, Math.min((x - a) / (b - a || 1), (c - x) / (c - b || 1)));
    const trapMF = (x, a, b, c, d) => Math.max(0, Math.min((x - a) / (b - a || 1), 1, (d - x) / (d - c || 1)));
    const gaussMF = (x, mu, sigma) => Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma || 1, 2)));

    const deleteVar = async () => {
        if (confirm(`Delete variable '${name}'?`)) {
            await fetch(`/variable/${name}`, { method: "DELETE" });
            refresh();
        }
    };

    // --- Drag & Drop Logic ---
    const handleMouseDown = (e) => {
        if (!e || !e.activeLabel) return;
        const xValue = parseFloat(e.activeLabel);

        // Find closest MF point
        let best = null;
        let minDist = (data.max - data.min) * 0.05; // 5% threshold

        Object.entries(data.terms).forEach(([tName, term]) => {
            const params = term.params;
            params.forEach((p, idx) => {
                const d = Math.abs(p - xValue);
                if (d < minDist) {
                    minDist = d;
                    best = { termKey: tName, index: idx, type: term.type };
                }
            });
        });

        if (best) {
            setDragging({ ...best, originalParams: [...data.terms[best.termKey].params] });
        }
    };

    const handleMouseMove = (e) => {
        if (!dragging || !e || !e.activeLabel) return;
        const newVal = parseFloat(e.activeLabel);

        // Local update for responsiveness
        const updatedParams = [...dragging.originalParams];
        updatedParams[dragging.index] = newVal;

        // Optional: Enforce ordering for Tri/Trap
        if (dragging.type.includes("Triangular") || dragging.type === "tri") {
            updatedParams.sort((a, b) => a - b);
        } else if (dragging.type.includes("Trapezoidal") || dragging.type === "trap") {
            updatedParams.sort((a, b) => a - b);
        }

        // Trigger a temporary UI update by modifying the local data projection
        // (This is computationally heavy if done every pixel, but Recharts is fast enough)
        // For efficiency, we just store the current drag state and re-render.
        setDragging(prev => ({ ...prev, currentParams: updatedParams }));
    };

    const handleMouseUp = async () => {
        if (!dragging || !dragging.currentParams) {
            setDragging(null);
            return;
        }

        // Commit to backend
        const t = data.terms[dragging.termKey];
        await fetch(`/variable/${name}/term`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: dragging.termKey,
                type: t.type.toLowerCase().includes("tri") ? "tri" :
                    t.type.toLowerCase().includes("trap") ? "trap" : "gauss",
                params: dragging.currentParams
            })
        });

        setDragging(null);
        refresh();
        fetchPlotData();
    };

    // Merge dragging state into display
    const currentData = useMemo(() => {
        if (!dragging || !dragging.currentParams) return plotData;

        const { termKey, type, currentParams } = dragging;
        const p = currentParams;

        return plotData.map(row => {
            const newRow = { ...row };
            const x = row.x;

            let val = 0;
            if (type.includes("Triangular") || type === "tri") val = triMF(x, p[0], p[1], p[2]);
            else if (type.includes("Trapezoidal") || type === "trap") val = trapMF(x, p[0], p[1], p[2], p[3]);
            else if (type.includes("Gaussian") || type === "gauss") val = gaussMF(x, p[0], p[1]);

            newRow[termKey] = val;
            return newRow;
        });
    }, [plotData, dragging]);

    return (
        <div className="flex flex-col h-full">
            {/* Toolbar */}
            <div className="h-16 border-b border-zinc-200 flex items-center justify-between px-8 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.01)]">
                <div className="flex items-center gap-6">
                    <h2 className="text-xl font-black text-puc-red tracking-tight">{name}</h2>
                    <div className="flex items-center gap-2">
                        <span className={`text-[9px] uppercase font-black tracking-widest px-2.5 py-1 rounded-full border ${data.role === 'input' ? 'bg-puc-red/5 text-puc-red border-puc-red/10' : 'bg-emerald-50 text-emerald-600 border-emerald-100'}`}>
                            {data.role} Variable
                        </span>
                        <div className="h-4 w-px bg-zinc-200 mx-1"></div>
                        <span className="text-[11px] text-puc-gray font-medium">
                            Range: <span className="text-zinc-900 font-bold">{data.min}</span> to <span className="text-zinc-900 font-bold">{data.max}</span>
                        </span>
                    </div>
                </div>
                <Button variant="danger" size="sm" icon={Icons.Trash} onClick={deleteVar}>Delete Variable</Button>
            </div>

            {/* Split View: Chart Top, Details Bottom */}
            <div className="flex-1 flex flex-col p-8 overflow-hidden space-y-6 max-w-6xl mx-auto w-full">
                {/* Chart Area */}
                <div className="flex-1 bg-white border border-zinc-200 rounded-xl relative min-h-0 flex flex-col shadow-sm">
                    <div className="absolute top-4 right-4 flex flex-wrap gap-2 pointer-events-none z-10 justify-end max-w-lg">
                        {Object.keys(data.terms).map((t, i) => (
                            <div key={t} className={`pointer-events-auto flex items-center gap-2 border px-3 py-1.5 rounded-lg text-[11px] shadow-sm cursor-pointer transition-all ${editingTerm === t ? 'bg-puc-red/5 border-puc-red/30 ring-2 ring-puc-red/10' : 'bg-white border-zinc-200 text-puc-gray hover:border-zinc-300'}`}
                                onClick={() => startEditTerm(t)}>
                                <span className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }}></span>
                                <span className="font-bold text-zinc-700">{t}</span>
                                <button onClick={(e) => { e.stopPropagation(); deleteTerm(t); }} className="ml-1 hover:text-puc-red text-zinc-300 transition-colors"><Icons.X /></button>
                            </div>
                        ))}
                    </div>
                    <div className="flex-1 min-h-0 p-4 cursor-crosshair">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart
                                data={currentData}
                                margin={{ top: 30, right: 30, bottom: 10, left: 10 }}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                <XAxis dataKey="x" stroke="#cbd5e1" tick={{ fontSize: 10, fontWeight: 600 }} tickFormatter={v => v.toFixed(1)} axisLine={false} tickLine={false} />
                                <YAxis hide domain={[0, 1]} />
                                <Tooltip
                                    contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '11px', fontWeight: 600 }}
                                    itemStyle={{ padding: '0px' }}
                                    active={!dragging}
                                />
                                {Object.keys(data.terms).map((t, i) => (
                                    <Area key={t} type="monotone" dataKey={t} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={dragging?.termKey === t ? 0.2 : 0.06} strokeWidth={dragging?.termKey === t ? 4 : 3} activeDot={{ r: 5, strokeWidth: 2, stroke: '#fff' }} />
                                ))}
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Properties Panel (Add/Edit Term) */}
                <div className={`h-auto shrink-0 border rounded-xl p-6 shadow-sm transition-colors ${editingTerm ? 'bg-amber-50/50 border-amber-200' : 'bg-white border-zinc-200'}`}>
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-4">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${editingTerm ? 'bg-amber-100 text-amber-700' : 'bg-puc-red/5 text-puc-red'}`}><Icons.Plus /></div>
                            <div>
                                <h3 className="text-sm font-black text-zinc-800 tracking-tight">{editingTerm ? `Editing: ${editingTerm}` : 'Add Membership Function'}</h3>
                                <p className="text-[11px] text-puc-gray font-medium">{editingTerm ? 'Modify the parameters and click Update Set' : 'Define a new fuzzy set for this variable'}</p>
                            </div>
                        </div>
                        {editingTerm && <button onClick={cancelEdit} className="text-[11px] font-bold text-zinc-400 hover:text-puc-red transition-colors">Cancel Edit</button>}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-end">
                        <div className="md:col-span-3"><Input label="Set Name" placeholder="e.g. Medium" value={termForm.name} onChange={e => setTermForm({ ...termForm, name: e.target.value })} disabled={!!editingTerm} /></div>
                        <div className="md:col-span-3">
                            <Select label="Function Type" value={termForm.type} onChange={e => setTermForm({ ...termForm, type: e.target.value })}>
                                <option value="tri">Triangular (a, b, c)</option>
                                <option value="trap">Trapezoidal (a, b, c, d)</option>
                                <option value="gauss">Gaussian (μ, σ)</option>
                            </Select>
                        </div>
                        <div className="md:col-span-4"><Input label="Parameters" placeholder="0.0, 5.0, 10.0" value={termForm.params} onChange={e => setTermForm({ ...termForm, params: e.target.value })} /></div>
                        <div className="md:col-span-2"><Button className="w-full" size="lg" variant={editingTerm ? 'default' : 'default'} onClick={addTerm}>{editingTerm ? 'Update Set' : 'Add Set'}</Button></div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ----------------------------------------------------------------------
// View: Rules
// ----------------------------------------------------------------------
function RulesView({ config, refresh }) {
    const [antecedents, setAntecedents] = useState([{ var: "", term: "" }]);
    const [consequent, setConsequent] = useState({ var: "", term: "" });

    const addRule = async () => {
        const payload = {};
        antecedents.forEach(a => { if (a.var && a.term) payload[a.var] = a.term; });
        if (!consequent.var || !consequent.term) return;
        await fetch("/rule", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ antecedents: payload, consequent: { [consequent.var]: consequent.term } })
        });
        refresh();
    };

    const deleteRule = async (i) => {
        await fetch(`/rule/${i}`, { method: "DELETE" }); refresh();
    };

    const inputs = Object.keys(config.variables).filter(v => config.variables[v].role === 'input');
    const outputs = Object.keys(config.variables).filter(v => config.variables[v].role === 'output');

    return (
        <div className="flex flex-col h-full max-w-6xl mx-auto w-full p-8 gap-8 overflow-y-auto custom-scrollbar">
            {/* Rule Builder Bar */}
            <div className="bg-white border border-zinc-200 rounded-xl p-8 flex flex-col gap-6 shadow-sm">
                <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-lg bg-puc-red/5 flex items-center justify-center text-puc-red font-black">IF</div>
                    <div>
                        <h3 className="text-sm font-black text-zinc-800 tracking-tight leading-tight uppercase tracking-[0.1em]">Knowledge Base Editor</h3>
                        <p className="text-[11px] text-puc-gray font-medium">Construct conditional logic using defined variables</p>
                    </div>
                </div>

                <div className="flex flex-col gap-4">
                    {/* Antecedents List */}
                    {antecedents.map((ant, i) => (
                        <div key={i} className="flex items-center gap-4 animate-in fade-in slide-in-from-left-2">
                            <div className={`w-12 text-right text-[10px] font-black tracking-widest ${i === 0 ? 'text-puc-red' : 'text-zinc-400'}`}>{i === 0 ? "IF" : "AND"}</div>

                            <div className="flex-1 max-w-sm">
                                <Select value={ant.var} onChange={e => { const c = [...antecedents]; c[i].var = e.target.value; c[i].term = ""; setAntecedents(c); }}>
                                    <option value="">Select Antecedent...</option>
                                    {inputs.map(v => <option key={v} value={v}>{v}</option>)}
                                </Select>
                            </div>

                            <span className="text-puc-gray text-[11px] font-black italic tracking-wider">IS</span>

                            <div className="flex-1 max-w-sm">
                                <Select value={ant.term} disabled={!ant.var} onChange={e => { const c = [...antecedents]; c[i].term = e.target.value; setAntecedents(c); }}>
                                    <option value="">Select Fuzzy Set...</option>
                                    {ant.var && Object.keys(config.variables[ant.var].terms).map(t => <option key={t} value={t}>{t}</option>)}
                                </Select>
                            </div>

                            {i > 0 ? (
                                <button onClick={() => setAntecedents(antecedents.filter((_, x) => x !== i))} className="p-2 text-zinc-300 hover:text-rose-500 transition-colors bg-zinc-50 rounded-lg"><Icons.Trash /></button>
                            ) : <div className="w-9"></div>}
                        </div>
                    ))}

                    {/* Add Condition Button */}
                    <div className="flex items-center gap-4">
                        <div className="w-12"></div>
                        <button onClick={() => setAntecedents([...antecedents, { var: "", term: "" }])} className="text-[11px] text-puc-red hover:text-[#a3442a] font-black uppercase tracking-widest flex items-center gap-2 px-2 py-1 transition-colors">
                            <Icons.Plus /> <span>Append Condition</span>
                        </button>
                    </div>
                </div>

                <div className="h-px bg-zinc-100 w-full"></div>

                {/* Consequent */}
                <div className="flex items-center gap-4">
                    <div className="w-12 text-right text-[10px] font-black tracking-widest text-[#059669]">THEN</div>
                    <div className="flex-1 max-w-sm">
                        <Select value={consequent.var} onChange={e => setConsequent(p => ({ ...p, var: e.target.value }))}>
                            <option value="">Select Consequent...</option>
                            {outputs.map(v => <option key={v} value={v}>{v}</option>)}
                        </Select>
                    </div>
                    <span className="text-puc-gray text-[11px] font-black italic tracking-wider">IS</span>
                    <div className="flex-1 max-w-sm">
                        <Select value={consequent.term} disabled={!consequent.var} onChange={e => setConsequent(p => ({ ...p, term: e.target.value }))}>
                            <option value="">Select Target Set...</option>
                            {consequent.var && Object.keys(config.variables[consequent.var].terms).map(t => <option key={t} value={t}>{t}</option>)}
                        </Select>
                    </div>
                    <div className="w-9"></div>
                </div>

                <div className="flex justify-end mt-4 pt-4 border-t border-zinc-50">
                    <Button onClick={addRule} variant="primary" size="lg" icon={Icons.Check}>Register Rule</Button>
                </div>
            </div>

            {/* Rule List */}
            <div className="flex-1 bg-white border border-zinc-200 rounded-xl overflow-hidden shadow-sm flex flex-col min-h-[350px]">
                <div className="px-6 py-4 border-b border-zinc-100 flex items-center justify-between bg-zinc-50/50">
                    <h4 className="text-[10px] font-black text-puc-gray uppercase tracking-[0.2em]">Active Ruleset</h4>
                    <span className="text-[10px] font-bold text-zinc-500">{config.rules.length} Rules Defined</span>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left text-xs">
                        <thead className="bg-white text-zinc-400 font-black uppercase tracking-widest sticky top-0 border-b border-zinc-100 z-10">
                            <tr>
                                <th className="px-4 py-4 text-[9px] w-12">#</th>
                                <th className="px-4 py-4 text-[9px]">Logical Statement</th>
                                <th className="px-4 py-4 text-[9px] w-24 text-right">Operations</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-50">
                            {config.rules.map((rule, i) => (
                                <tr key={i} className="hover:bg-zinc-50/80 group transition-colors">
                                    <td className="px-4 py-4 text-[11px] text-zinc-400 font-mono font-bold align-top">{i}</td>
                                    <td className="px-4 py-4">
                                        <div className="flex flex-wrap gap-y-2 items-center text-[13px] leading-none">
                                            <span className="font-black text-puc-red/80 mr-3 text-[10px] tracking-widest">IF</span>
                                            {Object.entries(rule.antecedents).map(([v, t], j) => (
                                                <span key={j} className="mr-4 flex items-center">
                                                    {j > 0 && <span className="text-zinc-400 mr-4 text-[10px] font-black">AND</span>}
                                                    <span className="text-zinc-500 mr-2 font-medium">{v}</span>
                                                    <span className="text-zinc-400 mr-2 italic font-serif">is</span>
                                                    <span className="text-zinc-900 font-bold bg-white border border-zinc-200 px-2 py-1 rounded shadow-sm">{t}</span>
                                                </span>
                                            ))}
                                            <div className="w-8 h-px bg-zinc-200 mx-2"></div>
                                            <span className="font-black text-emerald-600/80 mr-3 text-[10px] tracking-widest">THEN</span>
                                            <span className="text-zinc-500 mr-2 font-medium">{Object.keys(rule.consequent)[0]}</span>
                                            <span className="text-zinc-400 mr-2 italic font-serif">is</span>
                                            <span className="text-emerald-700 font-black bg-emerald-50 px-2.5 py-1 rounded border border-emerald-100">{Object.values(rule.consequent)[0]}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-5 text-right">
                                        <button onClick={() => deleteRule(i)} className="text-zinc-300 hover:text-puc-red opacity-0 group-hover:opacity-100 transition-all p-2 bg-zinc-50 rounded-lg hover:shadow-sm"><Icons.Trash /></button>
                                    </td>
                                </tr>
                            ))}
                            {config.rules.length === 0 && <tr><td colSpan="2" className="px-6 py-12 text-center text-zinc-400 font-medium italic">No rules defined in system.</td></tr>}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

// ----------------------------------------------------------------------
// View: Simulations
// ----------------------------------------------------------------------

function SimulateView({ config, refresh }) {
    const [scenario, setScenario] = useState("custom");
    const [status, setStatus] = useState("idle");
    const [error, setError] = useState(null);
    const [loadingPreset, setLoadingPreset] = useState(false);

    const build = async () => {
        setStatus("building"); setError(null);
        try { const res = await fetch("/build", { method: "POST" }); if (!res.ok) throw await res.json(); setStatus("ready"); }
        catch (e) { setError(e.detail); setStatus("idle"); }
    };

    const loadPreset = async () => {
        if (scenario === "custom") return;
        setLoadingPreset(true);
        try {
            await fetch(`/preset/${scenario}`, { method: "POST" });
            await refresh();
            alert(`FIS configured for ${scenario} scenario.`);
        } finally {
            setLoadingPreset(false);
        }
    };

    if (status === "idle") {
        return (
            <div className="flex h-full items-center justify-center p-8 bg-zinc-50">
                <div className="text-center p-12 bg-white border border-zinc-200 rounded-3xl shadow-xl max-w-md">
                    <div className="w-20 h-20 bg-puc-red/10 rounded-full flex items-center justify-center mx-auto mb-8 text-puc-red border border-puc-red/20 shadow-inner">
                        <div className="scale-150"><Icons.Power /></div>
                    </div>
                    <h2 className="text-2xl font-black text-zinc-900 mb-3 tracking-tight">System Idle</h2>
                    <p className="text-zinc-500 text-sm mb-8 font-medium leading-relaxed">Engage the control engine and select a physics scenario to begin real-time fuzzy inference.</p>
                    <div className="flex flex-col gap-3">
                        <Button size="lg" variant="primary" onClick={build} icon={Icons.Power}>Initialize Engine</Button>
                    </div>
                    {error && <div className="mt-6 p-3 bg-rose-50 border border-rose-100 rounded-lg text-rose-500 text-[11px] font-bold uppercase tracking-wider">{error}</div>}
                </div>
            </div>
        );
    }

    const scenarios = [
        { id: 'custom', label: 'Custom FIS Tester', group: 'General' },
        { id: 'dc_motor', label: 'DC Motor Speed (Exercise)', group: 'Classic Control' },
        { id: 'tank_broken', label: 'Broken Tank (Debug Exercise)', group: 'Classic Control' },
        { id: 'tank', label: 'Water Tank (sltank)', group: 'Classic Control' },
        { id: 'crane', label: 'Gantry Crane', group: 'Classic Control' },
        { id: 'robot_linear', label: 'Robot: Linear Vel.', group: 'Diff-Drive Robot' },
        { id: 'robot_angular', label: 'Robot: Angular Vel.', group: 'Diff-Drive Robot' },
        { id: 'robot_nav', label: 'Robot: Navigation', group: 'Diff-Drive Robot' }
    ];

    const groups = [...new Set(scenarios.map(s => s.group))];

    return (
        <div className="flex h-full w-full">
            <aside className="w-64 border-r border-zinc-200 bg-white p-6 shrink-0 flex flex-col gap-4 overflow-y-auto">
                {groups.map(group => (
                    <div key={group}>
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-[0.2em] mb-3">{group}</h3>
                        <div className="space-y-1.5 mb-4">
                            {scenarios.filter(s => s.group === group).map(s => (
                                <button key={s.id} onClick={() => setScenario(s.id)}
                                    className={`w-full text-left px-4 py-2.5 rounded-xl text-xs font-bold transition-all border ${scenario === s.id ? "bg-puc-red text-white shadow-md border-puc-red" : "bg-transparent text-puc-gray border-transparent hover:bg-zinc-50 hover:text-zinc-900"}`}>
                                    {s.label}
                                </button>
                            ))}
                        </div>
                    </div>
                ))}

                {scenario !== "custom" && (
                    <div className="p-4 bg-zinc-50 rounded-2xl border border-zinc-100 flex flex-col gap-3">
                        <p className="text-[10px] text-zinc-500 font-bold leading-tight">Mismatched Variables?</p>
                        <p className="text-[9px] text-zinc-400">Load the official FIS variables and base rules for this scenario.</p>
                        <Button variant="secondary" size="sm" onClick={loadPreset} disabled={loadingPreset}>
                            {loadingPreset ? "Loading..." : "Setup Scenario FIS"}
                        </Button>
                    </div>
                )}

                <div className="mt-auto pt-4 border-t border-zinc-100">
                    <Button variant="ghost" size="sm" className="w-full" onClick={() => setStatus('idle')}>Disconnect Engine</Button>
                </div>
            </aside>

            <main className="flex-1 min-w-0 bg-puc-stone/20 relative overflow-y-auto">
                {scenario === 'custom' && <CustomSimulation config={config} />}
                {scenario === 'dc_motor' && <DCMotorSimulation config={config} />}
                {(scenario === 'tank' || scenario === 'tank_broken') && <TankSimulation config={config} />}
                {scenario === 'crane' && <CraneSimulation config={config} />}
                {scenario === 'robot_linear' && <RobotLinearSim config={config} />}
                {scenario === 'robot_angular' && <RobotAngularSim config={config} />}
                {scenario === 'robot_nav' && <RobotNavSim config={config} />}
            </main>
        </div>
    );
}

// --- Simulation Step Inspector ---
function StepInspector({ data }) {
    if (!data) return null;
    
    // Process rule chart data
    const chartData = data.aggregated_output_shape.x.map((x, i) => {
        const row = { x, aggregated: data.aggregated_output_shape.y[i] };
        data.rules_triggered.forEach(r => {
            row[`R${r.index}`] = r.clipped_shape[i];
        });
        return row;
    });

    return (
        <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 flex flex-col gap-6 w-full mt-8">
            <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest">Inference Inspector</h3>
            
            {/* Fuzzification */}
            <div>
                <h4 className="text-xs font-bold text-zinc-700 mb-3">Input Fuzzification</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(data.input_memberships).map(([varName, terms]) => (
                        <div key={varName} className="bg-zinc-50 p-3 rounded-xl border border-zinc-100">
                            <span className="text-[10px] font-black uppercase text-zinc-500">{varName}</span>
                            <div className="mt-2 space-y-2">
                                {Object.entries(terms).map(([term, val]) => (
                                    <div key={term} className="flex items-center gap-2">
                                        <span className="w-20 text-[10px] font-bold text-zinc-600 truncate">{term}</span>
                                        <div className="flex-1 h-2 bg-zinc-200 rounded-full overflow-hidden">
                                            <div className="h-full bg-puc-red/60 rounded-full transition-all duration-200" style={{ width: `${val * 100}%` }}></div>
                                        </div>
                                        <span className="w-8 text-[10px] text-right font-mono text-zinc-500">{val.toFixed(2)}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Rule Contributions */}
            {data.rules_triggered.length > 0 && (
                <div>
                    <h4 className="text-xs font-bold text-zinc-700 mb-3">Fired Rules</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                        {data.rules_triggered.map((r, i) => (
                            <div key={i} className="bg-zinc-50 p-3 rounded-xl border border-zinc-100 flex items-center justify-between">
                                <div className="flex flex-col gap-1.5 flex-1 pr-4">
                                    <span className="text-[10px] font-black text-puc-red bg-puc-red/5 px-2 py-0.5 rounded w-max">R{r.index}</span>
                                    <p className="text-[11px] text-zinc-600 font-mono leading-relaxed">{r.description}</p>
                                </div>
                                <div className="flex flex-col items-end gap-1 w-24">
                                    <span className="text-[10px] font-bold text-zinc-500">Strength: {r.strength.toFixed(3)}</span>
                                    <div className="w-full h-1.5 bg-zinc-200 rounded-full overflow-hidden">
                                        <div className="h-full rounded-full transition-all duration-200" style={{ width: `${r.strength * 100}%`, backgroundColor: COLORS[i % COLORS.length] }}></div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Rule Output Chart */}
            <div className="h-56 mt-2">
                <h4 className="text-xs font-bold text-zinc-700 mb-3">Aggregated Output & Rule Contributions</h4>
                <div className="bg-zinc-50 p-4 rounded-xl border border-zinc-100 h-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                            <XAxis dataKey="x" stroke="#cbd5e1" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => v.toFixed(1)} />
                            <YAxis domain={[0, 1]} stroke="#cbd5e1" tick={{ fontSize: 10, fill: '#64748b' }} width={30} />
                            {data.rules_triggered.map((r, i) => (
                                <Area key={r.index} type="linear" dataKey={`R${r.index}`} 
                                    stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={0.2} strokeWidth={2} isAnimationActive={false} />
                            ))}
                            <Area type="linear" dataKey="aggregated" stroke="#1e293b" strokeWidth={3} fill="none" isAnimationActive={false} />
                            <ReferenceLine x={data.output} stroke="#8C3A22" strokeWidth={2} strokeDasharray="4 4" label={{ position: 'top', value: `Centroid = ${data.output.toFixed(3)}`, fill: '#8C3A22', fontSize: 11, fontWeight: 'bold' }} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

// --- DC Motor Simulation ---
function DCMotorSimulation({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({ rpm: 0, setpoint: 100, voltage: 0, history: [], inspectData: null });
    const physicsRef = useRef({ rpm: 0, voltage: 0, prevRpm: 0, angle: 0 });

    const updatePhysics = () => {
        const dt = 0.02; // 20ms
        const { rpm, voltage } = physicsRef.current;
        // Motor dynamics: dw/dt = -0.5*w + 10*V
        const wDot = -0.5 * rpm + 10 * voltage;
        physicsRef.current.rpm = rpm + wDot * dt;
        physicsRef.current.angle += physicsRef.current.rpm * (Math.PI / 30) * dt;
    };

    const syncFIS = async () => {
        const error = state.setpoint - physicsRef.current.rpm;
        const errorDot = (error - (state.setpoint - physicsRef.current.prevRpm)) / 0.1;
        physicsRef.current.prevRpm = physicsRef.current.rpm;

        try {
            const res = await fetch("/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { error: error, error_dot: errorDot } })
            });
            if (!res.ok) return;
            const data = await res.json();
            const voltage = data.output || 0;
            physicsRef.current.voltage = Math.max(-12, Math.min(12, voltage));
            
            setState(s => ({
                ...s, 
                rpm: physicsRef.current.rpm, 
                voltage: physicsRef.current.voltage,
                history: [...s.history.slice(-150), { t: Date.now(), rpm: physicsRef.current.rpm, sp: s.setpoint }],
                inspectData: data
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 20);
        const fisLoop = setInterval(syncFIS, 100);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.setpoint]);

    // Render Canvas Loop
    useEffect(() => {
        if (!canvasRef.current) return;
        let animationId;
        const ctx = canvasRef.current.getContext('2d');
        const W = 400, H = 350;

        const render = () => {
            ctx.clearRect(0, 0, W, H);
            
            ctx.fillStyle = '#e2e8f0';
            ctx.fillRect(120, 200, 110, 20);
            
            ctx.fillStyle = '#94a3b8';
            ctx.beginPath();
            ctx.roundRect(100, 100, 150, 100, 10);
            ctx.fill();

            ctx.fillStyle = '#64748b';
            ctx.fillRect(120, 100, 110, 10);
            ctx.fillRect(120, 190, 110, 10);
            
            const cx = 250, cy = 150;
            ctx.fillStyle = '#cbd5e1';
            ctx.fillRect(cx, cy - 8, 80, 16);
            
            ctx.save();
            ctx.translate(cx + 80, cy);
            ctx.rotate(physicsRef.current.angle);
            
            ctx.beginPath();
            ctx.arc(0, 0, 40, 0, 2 * Math.PI);
            ctx.fillStyle = '#f8fafc';
            ctx.fill();
            ctx.lineWidth = 6;
            ctx.strokeStyle = '#475569';
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(40, 0);
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 4;
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = '#1e293b';
            ctx.font = 'bold 24px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(`V = ${physicsRef.current.voltage.toFixed(1)}V`, 175, 250);

            animationId = requestAnimationFrame(render);
        };
        render();

        return () => cancelAnimationFrame(animationId);
    }, []);

    return (
        <div className="h-full flex flex-col items-center p-8 gap-8 overflow-y-auto custom-scrollbar">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl shrink-0">
                <div className="flex flex-col items-center bg-white p-8 rounded-3xl shadow-xl border border-zinc-100">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">DC Motor Dynamics</h3>
                    <canvas ref={canvasRef} width={400} height={350} className="w-full h-auto max-w-[350px]" />
                </div>
                <div className="flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-64">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">RPM Tracking</h3>
                        <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={state.history}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <YAxis domain={[-200, 200]} stroke="#cbd5e1" fontSize={10} />
                                <Line type="monotone" dataKey="rpm" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} isAnimationActive={false} />
                                <Line type="monotone" dataKey="sp" stroke="#8C3A22" strokeWidth={1} strokeDasharray="4 4" dot={false} animationDuration={0} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 text-center">
                        <label className="block text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Setpoint (RPM)</label>
                        <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                            min="-200" max="200" step="1" value={state.setpoint} onChange={e => setState(s => ({ ...s, setpoint: parseFloat(e.target.value) }))} />
                        <div className="mt-2 text-xl font-black text-puc-red">{state.setpoint} RPM</div>
                    </div>
                </div>
            </div>
            <div className="w-full max-w-6xl shrink-0">
                <StepInspector data={state.inspectData} />
            </div>
        </div>
    );
}

// --- Custom FIS Tester ---
function CustomSimulation({ config }) {
    const [inputs, setInputs] = useState({});
    const [output, setOutput] = useState(null);
    const [inspectData, setInspectData] = useState(null);
    const [error, setError] = useState(null);

    const inputVars = Object.entries(config.variables || {}).filter(([, v]) => v.role === 'input');
    const outputVar = Object.entries(config.variables || {}).find(([, v]) => v.role === 'output');

    useEffect(() => {
        const init = {};
        inputVars.forEach(([name, v]) => { init[name] = (v.min + v.max) / 2; });
        setInputs(init);
    }, [config]);

    const simulate = async () => {
        setError(null);
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs })
            });
            if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
            const data = await res.json();
            setOutput(data.output);
            setInspectData(data);
        } catch (e) { setError(e.message); }
    };

    if (inputVars.length === 0) {
        return (
            <div className="flex h-full items-center justify-center p-8">
                <div className="text-center p-12 bg-white border border-zinc-200 rounded-3xl shadow-xl max-w-md">
                    <h2 className="text-xl font-black text-zinc-900 mb-3">No FIS Configured</h2>
                    <p className="text-zinc-500 text-sm">Build your Fuzzy Inference System in the IDE tab first, then come back here to test it.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col items-center justify-center p-8 gap-8">
            <div className="bg-white p-8 rounded-3xl shadow-xl border border-zinc-100 w-full max-w-2xl">
                <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-6">Custom FIS Tester</h3>
                <div className="space-y-6">
                    {inputVars.map(([name, v]) => (
                        <div key={name}>
                            <div className="flex justify-between mb-2">
                                <label className="text-xs font-bold text-zinc-700">{name}</label>
                                <span className="text-xs font-black text-puc-red">{(inputs[name] || 0).toFixed(3)}</span>
                            </div>
                            <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                                min={v.min} max={v.max} step={(v.max - v.min) / 200}
                                value={inputs[name] || 0}
                                onChange={e => setInputs(prev => ({ ...prev, [name]: parseFloat(e.target.value) }))} />
                            <div className="flex justify-between text-[9px] text-zinc-400 mt-1">
                                <span>{v.min}</span><span>{v.max}</span>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="mt-6">
                    <Button variant="primary" size="lg" className="w-full" onClick={simulate}>Run Inference</Button>
                </div>
            </div>
            {output !== null && (
                <div className="w-full max-w-4xl flex flex-col gap-6">
                    <div className="bg-white p-8 rounded-3xl shadow-xl border border-zinc-100">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">
                            Output: {outputVar ? outputVar[0] : 'Result'}
                        </h3>
                        <p className="text-5xl font-black text-puc-red">{output.toFixed(4)}</p>
                    </div>
                    <StepInspector data={inspectData} />
                </div>
            )}
            {error && <div className="p-4 bg-rose-50 border border-rose-100 rounded-xl text-rose-600 text-sm font-bold w-full max-w-4xl">{error}</div>}
        </div>
    );
}

// --- Tank Simulation (MathWorks sltank-style) ---
function TankSimulation({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({ level: 0.2, setpoint: 0.7, valvePos: 0.5, history: [], inspectData: null });
    const physicsRef = useRef({ level: 0.2, valvePos: 0.5, prevLevel: 0.2 });

    const updatePhysics = () => {
        const dt = 0.05;
        const { level, valvePos } = physicsRef.current;
        const inflow = valvePos * 0.12;
        const outflow = 0.035 * Math.sqrt(Math.max(0, level));
        const newLevel = Math.max(0, Math.min(1, level + (inflow - outflow) * dt));
        physicsRef.current.prevLevel = level;
        physicsRef.current.level = newLevel;
    };

    const syncFIS = async () => {
        const { level, prevLevel, valvePos } = physicsRef.current;
        const levelErr = state.setpoint - level;
        const rate = level - prevLevel;
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { "level": levelErr, "rate": rate } })
            });
            const data = await res.json();
            const valveRate = data.output || 0;
            physicsRef.current.valvePos = Math.max(0, Math.min(1, valvePos + valveRate * 0.1));
            setState(s => ({
                ...s, level: physicsRef.current.level, valvePos: physicsRef.current.valvePos,
                history: [...s.history.slice(-150), { t: Date.now(), level: physicsRef.current.level, sp: s.setpoint }],
                inspectData: data
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 20);
        const fisLoop = setInterval(syncFIS, 100);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.setpoint]);

    useEffect(() => {
        const ctx = canvasRef.current.getContext('2d');
        const w = 400, h = 350;
        ctx.clearRect(0, 0, w, h);
        // Tank body
        ctx.strokeStyle = '#475569'; ctx.lineWidth = 3;
        ctx.strokeRect(100, 30, 200, 280);
        // Water
        const waterH = state.level * 280;
        const grad = ctx.createLinearGradient(100, 310 - waterH, 100, 310);
        grad.addColorStop(0, '#60a5fa'); grad.addColorStop(1, '#2563eb');
        ctx.fillStyle = grad;
        ctx.fillRect(102, 310 - waterH, 196, waterH);
        // Setpoint line
        const spY = 310 - state.setpoint * 280;
        ctx.strokeStyle = '#8C3A22'; ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(85, spY); ctx.lineTo(315, spY); ctx.stroke();
        ctx.setLineDash([]);
        // Inflow pipe + valve
        ctx.strokeStyle = '#64748b'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(60, 50); ctx.lineTo(100, 50); ctx.stroke();
        const valveW = state.valvePos * 30;
        ctx.fillStyle = '#22c55e'; ctx.fillRect(65, 42, valveW, 16);
        ctx.strokeStyle = '#475569'; ctx.strokeRect(65, 42, 30, 16);
        // Outflow pipe
        ctx.strokeStyle = '#64748b';
        ctx.beginPath(); ctx.moveTo(300, 290); ctx.lineTo(340, 290); ctx.stroke();
        // Labels
        ctx.fillStyle = '#1e293b'; ctx.font = 'bold 11px Inter';
        ctx.fillText(`Level: ${(state.level * 100).toFixed(1)}%`, 110, 330);
        ctx.fillStyle = '#8C3A22';
        ctx.fillText(`SP: ${(state.setpoint * 100).toFixed(0)}%`, 240, 330);
        ctx.fillStyle = '#22c55e';
        ctx.fillText(`Valve: ${(state.valvePos * 100).toFixed(0)}%`, 60, 75);
    }, [state.level, state.valvePos, state.setpoint]);

    return (
        <div className="h-full flex flex-col items-center p-8 gap-8 overflow-y-auto custom-scrollbar">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl shrink-0">
                <div className="flex flex-col items-center bg-white p-8 rounded-3xl shadow-xl border border-zinc-100">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Water Tank — sltank Model</h3>
                    <canvas ref={canvasRef} width={400} height={350} className="w-full h-auto max-w-[350px]" />
                </div>
                <div className="flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-64">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Level Tracking</h3>
                        <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={state.history}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <YAxis domain={[0, 1]} stroke="#cbd5e1" fontSize={10} />
                                <Line type="monotone" dataKey="level" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} isAnimationActive={false} />
                                <Line type="monotone" dataKey="sp" stroke="#8C3A22" strokeWidth={1} strokeDasharray="4 4" dot={false} animationDuration={0} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 text-center">
                        <label className="block text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Setpoint</label>
                        <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                            min="0" max="1" step="0.01" value={state.setpoint} onChange={e => setState(s => ({ ...s, setpoint: parseFloat(e.target.value) }))} />
                        <div className="mt-2 text-xl font-black text-puc-red">{(state.setpoint * 100).toFixed(0)}%</div>
                    </div>
                </div>
            </div>
            <div className="w-full max-w-6xl shrink-0">
                <StepInspector data={state.inspectData} />
            </div>
        </div>
    );
}

// --- Crane Simulation (Improved 2D) ---
function CraneSimulation({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({ x: 0, theta: 0.05, target: 5.0, history: [] });
    const physicsRef = useRef({ x: 0, v: 0, theta: 0.05, omega: 0, force: 0 });

    const updatePhysics = () => {
        const dt = 0.01;
        const M = 2.0, m = 0.5, L = 2.0, g = 9.81, b = 0.1;
        const { x, v, theta, omega, force } = physicsRef.current;
        const sin_t = Math.sin(theta), cos_t = Math.cos(theta);
        const denom = M + m - m * cos_t * cos_t;
        const x_acc = (force + m * L * omega * omega * sin_t - m * g * sin_t * cos_t - b * v) / denom;
        const theta_acc = (g * sin_t * (M + m) - cos_t * (force + m * L * omega * omega * sin_t) - b * v * cos_t) / (L * denom);
        physicsRef.current.v += x_acc * dt;
        physicsRef.current.x += physicsRef.current.v * dt;
        physicsRef.current.omega += theta_acc * dt;
        physicsRef.current.theta += physicsRef.current.omega * dt;
    };

    const syncFIS = async () => {
        const { x, v, theta, omega } = physicsRef.current;
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { "PosErr": state.target - x, "Vel": v, "Angle": theta, "W": omega } })
            });
            const data = await res.json();
            physicsRef.current.force = data.output || 0;
            setState(s => ({
                ...s, x: physicsRef.current.x, theta: physicsRef.current.theta,
                history: [...s.history.slice(-150), { t: Date.now(), pos: x, angle: theta * 10 }]
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 10);
        const fisLoop = setInterval(syncFIS, 80);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.target]);

    useEffect(() => {
        const ctx = canvasRef.current.getContext('2d');
        const W = 700, H = 320;
        ctx.clearRect(0, 0, W, H);
        const scale = 50;
        const cx = W / 2, railY = 80;
        // Ground
        ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, railY + 60, W, H);
        // Gantry legs
        ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(60, railY); ctx.lineTo(60, railY + 60); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(W - 60, railY); ctx.lineTo(W - 60, railY + 60); ctx.stroke();
        // Rail beam
        ctx.strokeStyle = '#64748b'; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(40, railY); ctx.lineTo(W - 40, railY); ctx.stroke();
        // Target marker
        const targetPx = cx + (state.target - 5) * scale;
        ctx.fillStyle = 'rgba(140,58,34,0.15)';
        ctx.fillRect(targetPx - 8, railY - 15, 16, 35);
        ctx.strokeStyle = '#8C3A22'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(targetPx, railY - 15); ctx.lineTo(targetPx, railY + 200); ctx.stroke();
        ctx.setLineDash([]);
        // Trolley
        const trolleyPx = cx + (state.x - 5) * scale;
        ctx.fillStyle = '#1e293b';
        ctx.beginPath();
        ctx.roundRect(trolleyPx - 20, railY - 10, 40, 20, 4);
        ctx.fill();
        // Cable + Load
        const cableLen = 120;
        const loadPx = trolleyPx + Math.sin(state.theta) * cableLen;
        const loadPy = railY + 10 + Math.cos(state.theta) * cableLen;
        ctx.strokeStyle = '#334155'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(trolleyPx, railY + 10); ctx.lineTo(loadPx, loadPy); ctx.stroke();
        ctx.fillStyle = '#8C3A22';
        ctx.beginPath(); ctx.arc(loadPx, loadPy, 14, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 9px Inter'; ctx.textAlign = 'center';
        ctx.fillText('m', loadPx, loadPy + 3); ctx.textAlign = 'start';
        // Scale
        ctx.fillStyle = '#94a3b8'; ctx.font = '9px Inter';
        for (let i = 0; i <= 10; i++) {
            const px = cx + (i - 5) * scale;
            ctx.fillText(`${i}m`, px - 6, railY + 75);
            ctx.fillRect(px, railY + 60, 1, 8);
        }
    }, [state.x, state.theta, state.target]);

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 gap-6">
            <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 w-full max-w-4xl">
                <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Gantry Crane — Cart-Pendulum Dynamics</h3>
                <canvas ref={canvasRef} width={700} height={320} className="w-full h-auto" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-56">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-3">Telemetry</h3>
                    <ResponsiveContainer width="100%" height="80%">
                        <LineChart data={state.history}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                            <YAxis domain={[-10, 10]} stroke="#cbd5e1" fontSize={10} />
                            <Line type="monotone" dataKey="pos" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} name="Position" />
                            <Line type="monotone" dataKey="angle" stroke="#ef4444" strokeWidth={1.5} dot={false} animationDuration={0} name="Angle×10" />
                            <ReferenceLine y={state.target} stroke="#8C3A22" strokeDasharray="3 3" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 flex flex-col justify-center text-center">
                    <label className="block text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Target Position</label>
                    <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                        min="0" max="10" step="0.1" value={state.target} onChange={e => setState(s => ({ ...s, target: parseFloat(e.target.value) }))} />
                    <div className="mt-3 text-2xl font-black text-puc-red">{state.target.toFixed(1)}m</div>
                </div>
            </div>
        </div>
    );
}

// --- Robot: Linear Velocity Control ---
function RobotLinearSim({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({ pos: 0, vel: 0, target: 5, history: [] });
    const physicsRef = useRef({ pos: 0, vel: 0, power: 0 });

    const updatePhysics = () => {
        const dt = 0.01;
        const friction = 0.3;
        const maxAccel = 3.0;
        const { vel, power } = physicsRef.current;
        const accel = power * maxAccel - friction * vel;
        physicsRef.current.vel += accel * dt;
        physicsRef.current.pos += physicsRef.current.vel * dt;
    };

    const syncFIS = async () => {
        const { pos, vel } = physicsRef.current;
        const distErr = state.target - pos;
        const rate = vel;
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { "DistErr": distErr, "Rate": rate } })
            });
            const data = await res.json();
            physicsRef.current.power = data.output || 0;
            setState(s => ({
                ...s, pos: physicsRef.current.pos, vel: physicsRef.current.vel,
                history: [...s.history.slice(-150), { t: Date.now(), pos, vel, sp: s.target }]
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 10);
        const fisLoop = setInterval(syncFIS, 80);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.target]);

    useEffect(() => {
        const ctx = canvasRef.current.getContext('2d');
        const W = 700, H = 160;
        ctx.clearRect(0, 0, W, H);
        // Road
        ctx.fillStyle = '#f1f5f9'; ctx.fillRect(30, 60, 640, 40);
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
        ctx.setLineDash([8, 6]);
        ctx.beginPath(); ctx.moveTo(30, 80); ctx.lineTo(670, 80); ctx.stroke();
        ctx.setLineDash([]);
        // Scale
        ctx.fillStyle = '#94a3b8'; ctx.font = '9px Inter';
        for (let i = -5; i <= 15; i += 5) {
            const px = 350 + i * 30;
            ctx.fillText(`${i}m`, px - 6, 115);
        }
        // Target
        const targetPx = 350 + state.target * 30;
        ctx.fillStyle = 'rgba(140,58,34,0.3)';
        ctx.beginPath(); ctx.arc(targetPx, 80, 12, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#8C3A22'; ctx.font = 'bold 8px Inter'; ctx.textAlign = 'center';
        ctx.fillText('T', targetPx, 83); ctx.textAlign = 'start';
        // Robot (car shape)
        const robotPx = 350 + state.pos * 30;
        ctx.save();
        ctx.translate(robotPx, 80);
        // Body
        ctx.fillStyle = '#1e293b';
        ctx.beginPath(); ctx.roundRect(-14, -8, 28, 16, 4); ctx.fill();
        // Windshield
        ctx.fillStyle = '#60a5fa';
        ctx.beginPath(); ctx.roundRect(6, -5, 6, 10, 2); ctx.fill();
        // Wheels
        ctx.fillStyle = '#475569';
        ctx.fillRect(-12, -11, 7, 3);
        ctx.fillRect(-12, 8, 7, 3);
        ctx.fillRect(5, -11, 7, 3);
        ctx.fillRect(5, 8, 7, 3);
        ctx.restore();
        // Velocity arrow
        const arrowLen = state.vel * 20;
        if (Math.abs(arrowLen) > 2) {
            ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(robotPx, 55); ctx.lineTo(robotPx + arrowLen, 55); ctx.stroke();
        }
        ctx.fillStyle = '#1e293b'; ctx.font = 'bold 10px Inter';
        ctx.fillText(`v = ${state.vel.toFixed(2)} m/s`, 30, 145);
        ctx.fillText(`pos = ${state.pos.toFixed(2)} m`, 200, 145);
    }, [state.pos, state.vel, state.target]);

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 gap-6">
            <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 w-full max-w-4xl">
                <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Robot — Linear Velocity Control</h3>
                <canvas ref={canvasRef} width={700} height={160} className="w-full h-auto" />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-56">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-3">Position & Velocity</h3>
                    <ResponsiveContainer width="100%" height="80%">
                        <LineChart data={state.history}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                            <YAxis domain={[-5, 15]} stroke="#cbd5e1" fontSize={10} />
                            <Line type="monotone" dataKey="pos" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} />
                            <Line type="monotone" dataKey="vel" stroke="#22c55e" strokeWidth={1.5} dot={false} animationDuration={0} />
                            <Line type="monotone" dataKey="sp" stroke="#8C3A22" strokeWidth={1} strokeDasharray="4 4" dot={false} animationDuration={0} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 flex flex-col justify-center text-center">
                    <label className="block text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Target Distance</label>
                    <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                        min="-5" max="10" step="0.5" value={state.target} onChange={e => setState(s => ({ ...s, target: parseFloat(e.target.value) }))} />
                    <div className="mt-3 text-2xl font-black text-puc-red">{state.target}m</div>
                </div>
            </div>
        </div>
    );
}

// --- Robot: Angular Velocity Control ---
function RobotAngularSim({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({ omega: 0, theta: 0, targetOmega: 2.0, history: [] });
    const physicsRef = useRef({ omega: 0, theta: 0, steering: 0 });

    const updatePhysics = () => {
        const dt = 0.01;
        const inertia = 0.5;
        const friction = 0.2;
        const maxTorque = 5.0;
        const { omega, steering } = physicsRef.current;
        const torque = steering * maxTorque;
        const alpha = (torque - friction * omega) / inertia;
        physicsRef.current.omega += alpha * dt;
        physicsRef.current.theta += physicsRef.current.omega * dt;
    };

    const syncFIS = async () => {
        const { omega } = physicsRef.current;
        const angErr = state.targetOmega - omega;
        const prevOmega = state.omega;
        const angRate = omega - prevOmega;
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { "AngErr": angErr, "AngRate": angRate } })
            });
            const data = await res.json();
            physicsRef.current.steering = data.output || 0;
            setState(s => ({
                ...s, omega: physicsRef.current.omega, theta: physicsRef.current.theta,
                history: [...s.history.slice(-150), { t: Date.now(), omega, target: s.targetOmega }]
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 10);
        const fisLoop = setInterval(syncFIS, 80);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.targetOmega]);

    useEffect(() => {
        const ctx = canvasRef.current.getContext('2d');
        const W = 350, H = 350;
        const cx = W / 2, cy = H / 2;
        ctx.clearRect(0, 0, W, H);
        // Background circle
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, 120, 0, Math.PI * 2); ctx.stroke();
        // Gauge ticks
        for (let i = 0; i < 12; i++) {
            const a = (i / 12) * Math.PI * 2;
            ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx + Math.cos(a) * 110, cy + Math.sin(a) * 110);
            ctx.lineTo(cx + Math.cos(a) * 120, cy + Math.sin(a) * 120);
            ctx.stroke();
        }
        // Robot body (car shape)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(state.theta);
        // Car body
        ctx.fillStyle = '#1e293b';
        ctx.beginPath(); ctx.roundRect(-30, -16, 60, 32, 8); ctx.fill();
        // Windshield
        ctx.fillStyle = '#60a5fa';
        ctx.beginPath(); ctx.roundRect(12, -10, 12, 20, 3); ctx.fill();
        // Headlights
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath(); ctx.arc(28, -8, 3, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(28, 8, 3, 0, Math.PI * 2); ctx.fill();
        // Wheels
        ctx.fillStyle = '#475569';
        ctx.fillRect(-24, -22, 14, 6);
        ctx.fillRect(-24, 16, 14, 6);
        ctx.fillRect(10, -22, 14, 6);
        ctx.fillRect(10, 16, 14, 6);
        ctx.restore();
        ctx.fillStyle = '#1e293b'; ctx.font = 'bold 12px Inter'; ctx.textAlign = 'center';
        ctx.fillText(`ω = ${state.omega.toFixed(2)} rad/s`, cx, H - 20);
        ctx.fillText(`θ = ${(state.theta % (Math.PI * 2)).toFixed(2)} rad`, cx, H - 5);
        ctx.textAlign = 'start';
    }, [state.omega, state.theta]);

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 gap-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 flex flex-col items-center">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Robot — Angular Velocity Control</h3>
                    <canvas ref={canvasRef} width={350} height={350} className="w-full h-auto max-w-[300px]" />
                </div>
                <div className="flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-56">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-3">ω Tracking</h3>
                        <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={state.history}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <YAxis domain={[-5, 5]} stroke="#cbd5e1" fontSize={10} />
                                <Line type="monotone" dataKey="omega" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} />
                                <Line type="monotone" dataKey="target" stroke="#8C3A22" strokeWidth={1} strokeDasharray="4 4" dot={false} animationDuration={0} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 text-center">
                        <label className="block text-[10px] font-black text-puc-gray uppercase tracking-widest mb-4">Target ω (rad/s)</label>
                        <input type="range" className="w-full h-2 bg-zinc-100 rounded-lg appearance-none cursor-pointer accent-puc-red"
                            min="-5" max="5" step="0.1" value={state.targetOmega} onChange={e => setState(s => ({ ...s, targetOmega: parseFloat(e.target.value) }))} />
                        <div className="mt-3 text-2xl font-black text-puc-red">{state.targetOmega.toFixed(1)} rad/s</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// --- Robot: Full Navigation ---
function RobotNavSim({ config }) {
    const canvasRef = useRef(null);
    const [state, setState] = useState({
        x: 1, y: 1, theta: 0, targetX: 8, targetY: 8,
        trail: [], history: []
    });
    const physicsRef = useRef({ x: 1, y: 1, theta: 0, v: 0, omega: 0, prevDist: 0, prevHead: 0 });

    const updatePhysics = () => {
        const dt = 0.02;
        const { x, y, theta, v, omega } = physicsRef.current;
        physicsRef.current.x += v * Math.cos(theta) * dt;
        physicsRef.current.y += v * Math.sin(theta) * dt;
        physicsRef.current.theta += omega * dt;
    };

    const syncFIS = async () => {
        const { x, y, theta, prevDist, prevHead } = physicsRef.current;
        const dx = state.targetX - x;
        const dy = state.targetY - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const desiredAngle = Math.atan2(dy, dx);
        let headErr = desiredAngle - theta;
        while (headErr > Math.PI) headErr -= 2 * Math.PI;
        while (headErr < -Math.PI) headErr += 2 * Math.PI;
        const dDist = dist - prevDist;
        const dHead = headErr - prevHead;
        try {
            const res = await fetch("/simulate", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ inputs: { "DistErr": dist, "dDist": dDist, "HeadErr": headErr, "dHead": dHead } })
            });
            const data = await res.json();
            const vel = (data.output || 0) * 3;
            physicsRef.current.v = vel;
            physicsRef.current.omega = headErr * 2.0;
            physicsRef.current.prevDist = dist;
            physicsRef.current.prevHead = headErr;
            setState(s => ({
                ...s, x: physicsRef.current.x, y: physicsRef.current.y, theta: physicsRef.current.theta,
                trail: [...s.trail.slice(-300), { x: physicsRef.current.x, y: physicsRef.current.y }],
                history: [...s.history.slice(-150), { t: Date.now(), dist, headErr: headErr * (180 / Math.PI) }]
            }));
        } catch (e) { }
    };

    useEffect(() => {
        const physLoop = setInterval(updatePhysics, 20);
        const fisLoop = setInterval(syncFIS, 100);
        return () => { clearInterval(physLoop); clearInterval(fisLoop); };
    }, [state.targetX, state.targetY]);

    const handleCanvasClick = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const scaleX = 500 / rect.width;
        const scaleY = 500 / rect.height;
        const clickX = ((e.clientX - rect.left) * scaleX - 50) / 40;
        const clickY = (500 - 50 - (e.clientY - rect.top) * scaleY) / 40;
        setState(s => ({ ...s, targetX: Math.max(0, Math.min(10, clickX)), targetY: Math.max(0, Math.min(10, clickY)), trail: [] }));
        physicsRef.current.prevDist = 0;
        physicsRef.current.prevHead = 0;
    };

    useEffect(() => {
        const ctx = canvasRef.current.getContext('2d');
        const W = 500, H = 500;
        const ox = 50, oy = H - 50;
        const s = 40;
        ctx.clearRect(0, 0, W, H);
        // Grid
        ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            ctx.beginPath(); ctx.moveTo(ox + i * s, 10); ctx.lineTo(ox + i * s, oy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(ox, oy - i * s); ctx.lineTo(ox + 10 * s, oy - i * s); ctx.stroke();
        }
        // Axes
        ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ox, 10); ctx.lineTo(ox, oy); ctx.lineTo(ox + 10 * s + 10, oy); ctx.stroke();
        ctx.fillStyle = '#94a3b8'; ctx.font = '9px Inter';
        for (let i = 0; i <= 10; i += 2) {
            ctx.fillText(`${i}`, ox + i * s - 3, oy + 15);
            ctx.fillText(`${i}`, ox - 18, oy - i * s + 3);
        }
        // Trail
        if (state.trail.length > 1) {
            ctx.strokeStyle = 'rgba(59,130,246,0.3)'; ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(ox + state.trail[0].x * s, oy - state.trail[0].y * s);
            state.trail.forEach(p => ctx.lineTo(ox + p.x * s, oy - p.y * s));
            ctx.stroke();
        }
        // Target
        const targetPx = ox + state.targetX * s, targetPy = oy - state.targetY * s;
        ctx.fillStyle = 'rgba(140,58,34,0.2)';
        ctx.beginPath(); ctx.arc(targetPx, targetPy, 15, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#8C3A22';
        ctx.beginPath(); ctx.arc(targetPx, targetPy, 5, 0, Math.PI * 2); ctx.fill();
        // Robot (car shape)
        const rpx = ox + state.x * s, rpy = oy - state.y * s;
        ctx.save();
        ctx.translate(rpx, rpy);
        ctx.rotate(-state.theta);
        // Body
        ctx.fillStyle = '#1e293b';
        ctx.beginPath(); ctx.roundRect(-10, -6, 20, 12, 3); ctx.fill();
        // Windshield
        ctx.fillStyle = '#60a5fa';
        ctx.beginPath(); ctx.roundRect(4, -4, 4, 8, 1.5); ctx.fill();
        // Wheels
        ctx.fillStyle = '#475569';
        ctx.fillRect(-8, -8, 5, 2);
        ctx.fillRect(-8, 6, 5, 2);
        ctx.fillRect(3, -8, 5, 2);
        ctx.fillRect(3, 6, 5, 2);
        ctx.restore();
    }, [state.x, state.y, state.theta, state.targetX, state.targetY, state.trail]);

    return (
        <div className="h-full flex flex-col items-center justify-center p-6 gap-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-5xl">
                <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100">
                    <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-2">Robot Navigation — Click to Set Target</h3>
                    <canvas ref={canvasRef} width={500} height={500} className="w-full h-auto cursor-crosshair" onClick={handleCanvasClick} />
                </div>
                <div className="flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100 h-56">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-3">Distance & Heading Error</h3>
                        <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={state.history}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <YAxis stroke="#cbd5e1" fontSize={10} />
                                <Line type="monotone" dataKey="dist" stroke="#3b82f6" strokeWidth={2} dot={false} animationDuration={0} name="Distance" />
                                <Line type="monotone" dataKey="headErr" stroke="#ef4444" strokeWidth={1.5} dot={false} animationDuration={0} name="HeadErr°" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-3xl shadow-xl border border-zinc-100">
                        <h3 className="text-[10px] font-black text-puc-gray uppercase tracking-widest mb-3">State</h3>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                            <div className="bg-zinc-50 p-3 rounded-xl"><span className="text-puc-gray font-bold">X:</span> <span className="font-black">{state.x.toFixed(2)}m</span></div>
                            <div className="bg-zinc-50 p-3 rounded-xl"><span className="text-puc-gray font-bold">Y:</span> <span className="font-black">{state.y.toFixed(2)}m</span></div>
                            <div className="bg-zinc-50 p-3 rounded-xl"><span className="text-puc-gray font-bold">θ:</span> <span className="font-black">{(state.theta * 180 / Math.PI).toFixed(1)}°</span></div>
                            <div className="bg-zinc-50 p-3 rounded-xl"><span className="text-puc-gray font-bold">Target:</span> <span className="font-black">({state.targetX.toFixed(1)}, {state.targetY.toFixed(1)})</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
