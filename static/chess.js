const glyphs = {K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
const files = "abcdefgh";
let state = {
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    playerColor: "white",
    depth: 3,
    legalMoves: [],
    history: [],
    lastMove: null,
    selected: null,
    busy: false,
    status: {over: false, message: "Starting new game..."}
};
const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const toastEl = document.getElementById("toast");

function showToast(message) {
  toastEl.textContent = message; toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 2200);
}
async function api(path, body) {
  const response = await fetch(path, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
function parseFen(fen) {
  const rows = fen.split(" ")[0].split("/");
  const squares = [];
  rows.forEach(row => {
    for (const ch of row) {
      if (/[1-8]/.test(ch)) for(let i=0;i<Number(ch);i++) squares.push(".");
      else squares.push(ch);
    }
  });
  return squares;
}
function squareName(index) { return files[index%8] + (8-Math.floor(index/8)); }
function squareIndex(name) { return (8-Number(name[1]))*8 + files.indexOf(name[0]); }
function fenTurn() { return state.fen.split(" ")[1] === "w" ? "white" : "black"; }
function render() {
  const pieces = parseFen(state.fen);
  boardEl.innerHTML = "";
  const flipped = state.playerColor === "black";
  for(let display=0; display<64; display++) {
    const index = flipped ? 63-display : display;
    const row=Math.floor(index/8), col=index%8, piece=pieces[index];
    const sq=document.createElement("button");
    sq.type="button"; sq.className="square "+((row+col)%2===0?"light":"dark");
    sq.dataset.index=index;
    if(state.selected===index) sq.classList.add("selected");
    if(state.lastMove && (state.lastMove.from===index || state.lastMove.to===index)) sq.classList.add("last-move");
    const targets=state.selected===null?[]:state.legalMoves.filter(m=>m.slice(0,2)===squareName(state.selected)).map(m=>m.slice(2,4));
    if(targets.includes(squareName(index))) sq.classList.add(piece==="."?"target":"capture-target");
    if(col===0){const c=document.createElement("span");c.className="coord";c.textContent=8-row;sq.appendChild(c);}
    if(row===7){const c=document.createElement("span");c.className="coord";c.style.left="auto";c.style.right="4px";c.style.top="auto";c.style.bottom="3px";c.textContent=files[col];sq.appendChild(c);}
    if(piece!=="."){const p=document.createElement("span");p.className="piece "+(piece===piece.toUpperCase()?"white":"black");p.textContent=glyphs[piece];sq.appendChild(p);}
    sq.addEventListener("click",()=>onSquare(index));
    boardEl.appendChild(sq);
  }
  statusEl.textContent=state.busy?"Engine is thinking…":state.status?.message || "Ready";
  document.getElementById("depth-label").textContent="Depth "+state.depth;
  document.getElementById("depth-value").textContent="Depth "+state.depth;
  document.getElementById("human-label").textContent="Human · "+cap(state.playerColor);
  document.getElementById("bot-label").textContent="Engine · "+cap(state.playerColor==="white"?"black":"white");
  document.getElementById("turn-indicator").classList.toggle("active",!state.busy && fenTurn()===state.playerColor && !state.status?.over);
  renderHistory();
}
function cap(s){return s[0].toUpperCase()+s.slice(1);}
function renderHistory(){
  const el=document.getElementById("move-list");
  if(!state.history.length){el.innerHTML='<p class="empty-state">Your moves will appear here.</p>';return;}
  el.innerHTML="";
  for(let i=0;i<state.history.length;i+=2){
    const row=document.createElement("div");row.className="move-row";
    const no=document.createElement("span");no.className="move-no";no.textContent=(Math.floor(i/2)+1)+".";
    const a=document.createElement("span");a.className="move-san";a.textContent=state.history[i]?.san||state.history[i]?.uci||"";
    const b=document.createElement("span");b.className="move-san";b.textContent=state.history[i+1]?.san||state.history[i+1]?.uci||"";
    row.append(no,a,b);el.appendChild(row);
  }
  el.scrollTop=el.scrollHeight;
}
function sanFor(uci, beforeFen){
  // The backend supplies legal UCI moves; this lightweight display uses UCI notation.
  return uci;
}
async function refreshLegal(){
  const data=await api("/api/legal-moves",{fen:state.fen});
  state.legalMoves=data.legal_moves;
}
async function startGame(){
  state.busy=true;state.selected=null;state.history=[];state.lastMove=null;render();
  try{
    state.playerColor=document.getElementById("color-select").value;
    state.depth=Number(document.getElementById("depth-select").value);
    const data=await api("/api/new-game",{player_color:state.playerColor,depth:state.depth});
    state.fen=data.fen;state.status=data.status;state.legalMoves=data.legal_moves;
    if(data.moves) state.history.push(...data.moves);
    state.lastMove=data.bot_move?{from:squareIndex(data.bot_move.slice(0,2)),to:squareIndex(data.bot_move.slice(2,4))}:null;
  }catch(e){showToast(e.message);}
  state.busy=false;render();
}
async function onSquare(index){
  if(state.busy||state.status?.over||fenTurn()!==state.playerColor)return;
  const pieces=parseFen(state.fen), piece=pieces[index], name=squareName(index);
  if(state.selected===null){
    if(piece==="." || (state.playerColor==="white" ? piece!==piece.toUpperCase():piece!==piece.toLowerCase()))return;
    state.selected=index;render();return;
  }
  const from=squareName(state.selected);
  if(index===state.selected){state.selected=null;render();return;}
  let candidates=state.legalMoves.filter(m=>m.slice(0,2)===from&&m.slice(2,4)===name);
  if(candidates.length===0){
    if(piece!=="." && (state.playerColor==="white"?piece===piece.toUpperCase():piece===piece.toLowerCase())){state.selected=index;render();}
    else {state.selected=null;render();}
    return;
  }
  let move=candidates[0];
  if(candidates.length>1){
    const choice=prompt("Promote to: q (queen), r (rook), b (bishop), n (knight)","q");
    const promo=(choice||"q").toLowerCase();
    move=candidates.find(m=>m[4]===promo)||candidates[0];
  }
  state.busy=true;state.selected=null;render();
  const oldFen=state.fen;
  try{
    const data=await api("/api/move",{fen:oldFen,player_color:state.playerColor,depth:state.depth,move});
    state.fen=data.fen;state.status=data.status;state.legalMoves=data.legal_moves;
    if(data.moves) state.history.push(...data.moves);
    state.lastMove={from:index,to:squareIndex(name)};
  }catch(e){showToast(e.message);await refreshLegal();}
  state.busy=false;render();
}
document.getElementById("new-game").addEventListener("click",startGame);
document.getElementById("depth-select").addEventListener("change",e=>document.getElementById("depth-value").textContent="Depth "+e.target.value);
document.getElementById("copy-pgn").addEventListener("click",async()=>{
  const text=state.history.map((m,i)=>(i%2===0?`${Math.floor(i/2)+1}. `:"")+m.san).join(" ");
  try{await navigator.clipboard.writeText(text||"");showToast("Move list copied.");}catch(e){showToast("Clipboard unavailable in this browser.");}
});
(async()=>{try{const r=await fetch("/api/health");if(!r.ok)throw new Error();await startGame();}catch(e){statusEl.textContent="Could not connect to the ChessAI server.";}})();
