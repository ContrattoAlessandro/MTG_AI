// `mz lan`: the page of `mz web` at a table among friends on the local network.
//
// The page is unchanged; this script plugs into it through window.mzTable:
// - connect: views arrive over a WebSocket (long polling if it cannot connect);
// - lens: at a table of three or more, the page draws one opponent at a time, as in a duel.
//   A column of the opponents sits in the top-right corner; a click on one brings them in view;
// - afterRender: draws the seat column and the attack labels, and keeps the lobby up to date;
// - chooseDefender: fans out the players (and planeswalkers) a creature can attack;
// - openLobby: seats, names, decks and bots instead of the duel's deck picker.
// Commands go over HTTP; the server knows each browser's seat by its cookie.
"use strict";
(() => {
  // ---------- transport ----------
  // The server pushes this seat's view after every change; a closed socket reconnects, and
  // after a few failures the page falls back to long polling.
  let socket = null, failures = 0, raw = null, lastGame = null, shareUrl = null;
  function connect(accept) {
    const deliver = (next) => {
      raw = next;
      if (next.game !== lastGame && next.state) { lastGame = next.game; post("/api/mode", { mode: passMode }).catch(() => {}); }
      accept(next);
    };
    const open = async () => {
      let port;
      try {
        const config = await (await fetch("/api/config")).json();
        port = config.push_port; shareUrl = config.share_url;
      } catch (e) { return retry(); }
      socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${port}/`);
      let wire = null;  // the view as last received: a new connection starts with a whole view
      socket.onmessage = (e) => {
        failures = 0;
        const message = JSON.parse(e.data);
        wire = message.delta ? patch(wire, message.delta) : message;
        deliver(structuredClone(wire));  // the page may change what it is given
      };
      socket.onclose = (e) => { socket = null; if (e.code !== 4000) retry(); else open(); };
    };
    const retry = () => {
      failures++;
      setStatus("err", "Lost the connection to the host, retrying…");
      if (failures > 4) longPoll(deliver); else setTimeout(open, 600 * failures);
    };
    open();
  }
  async function longPoll(deliver) {
    for (let since = null; ;) {
      try {
        const next = await (await fetch("/api/state" + (since == null ? "" : "?since=" + since))).json();
        since = next.version; deliver(next);
      } catch (e) { setStatus("err", "Lost the connection to the host, retrying…"); await new Promise((r) => setTimeout(r, 1500)); }
    }
  }
  // After the first view the server sends only what changed (tabletop/push.py: delta): top-level
  // keys, the state's changed keys, and the log's sliding window as lines dropped and added.
  function patch(base, d) {
    const next = { ...base, ...d.set };
    if ("state" in d) next.state = d.state;
    else if (d.patch || d.log) {
      const state = { ...base.state, ...d.patch };
      if (d.log) state.log = base.state.log.slice(d.log.drop).concat(d.log.add);
      next.state = state;
    }
    return next;
  }
  // The socket was opened before the seat cookie existed: open it again to receive the seat's view.
  function reconnect() { if (socket) socket.close(4000, "seat changed"); }

  // ---------- the opponent in view ----------
  // The page follows whoever's turn it is (and the attacker when you block) unless the person
  // has chosen a player this turn; an eliminated player gives way to the next one.
  const focus = { id: null, manual: null, turnKey: null, blockFor: null };
  function chooseFocus(s, d, opponents) {
    const me = s.players.find((p) => p.side === "you");
    const key = s.turn + ":" + s.active_id;
    if (key !== focus.turnKey) { focus.turnKey = key; focus.manual = null; }
    const blocking = d && d.kind === "blockers" ? d.decision : null;
    if (blocking && focus.blockFor !== blocking) { focus.blockFor = blocking; focus.manual = null; }
    let id = focus.manual || (s.active_id && s.active_id !== me.id ? s.active_id : focus.id);
    const alive = opponents.filter((p) => !p.lost);
    if (!alive.some((p) => p.id === id)) id = (alive[0] || opponents[0]).id;
    focus.id = id;
    return opponents.find((p) => p.id === id);
  }
  function show(id) {
    focus.manual = id;
    if (raw) { view = lens(raw); render(); }
  }

  // The view as a duel between you and the opponent in view; `everyone` keeps the whole table.
  function lens(next) {
    const s = next.state;
    if (!s) return next;
    const players = s.players.map((p) => ({ ...p, deck: deckOf(next, p.name) }));
    if (players.length <= 2) return { ...next, state: { ...s, players } };
    const me = players.find((p) => p.side === "you");
    const shown = chooseFocus(s, next.decision, players.filter((p) => p !== me));
    const keep = new Set([me.id, shown.id]);
    const onTable = new Set(s.battlefield.filter((c) => keep.has(c.controller_id)).map((c) => c.id));
    // An aura or equipment of someone else stays with the creature it is attached to.
    const battlefield = s.battlefield.filter((c) => onTable.has(c.id) || (c.attached_to && onTable.has(c.attached_to)));
    return { ...next, state: { ...s, players: [me, shown], everyone: players, battlefield } };
  }
  function deckOf(next, name) {
    const seat = next.lobby && next.lobby.seats.find((x) => x.name === name);
    const known = seat && decks.find((d) => d.name === seat.deck);
    return known || (seat && seat.deck ? { name: seat.deck } : null);
  }

  // ---------- the seat column ----------
  // The opponents in turn order, one square each down the top-right corner of the table:
  // commander, name and life. A click brings that player's board in view; a glowing square
  // is a player the pending decision can act on.
  let compact = null;  // the column shows names and life only (remembered; read once the page's helpers exist)
  const SWORDS = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M4 4l11 11M20 4L9 15M13 17l4 4M11 17l-4 4M15 15l3-3M9 15l-3-3"/></svg>';
  function renderSeatMap(v) {
    if (compact === null) compact = load("lanCompact", false);
    let map = $("seatMap");
    const s = v.state, players = s && s.everyone;
    if (!players) { if (map) map.remove(); return; }
    if (!map) { map = h("div", { id: "seatMap", class: "seat-map", role: "tablist", "aria-label": "Opponents" }); $("table").append(map); }
    map.classList.toggle("compact", compact);
    const whole = raw.state, d = v.decision, shownId = s.players[1].id;
    const me = players.find((p) => p.side === "you");
    // Players the pending decision can act on (them, or one of their permanents).
    const wanted = new Set();
    if (d && !answered.has(d.decision)) {
      for (const o of d.options) for (const ref of [o.target, o.source]) {
        const owner = ref && (players.some((p) => p.id === ref) ? ref : controllerOf(ref));
        if (owner) wanted.add(owner);
      }
    }
    const attacking = new Set((whole.combat || []).filter((g) => g.defending_player === me.id).flatMap((g) => g.attackers).map(controllerOf));
    // The engine lists the players in turn order, starting with you.
    const toggle = h("button", { class: "sm-toggle", title: compact ? "Show the opponents' squares" : "Shrink the opponents to name and life",
      "aria-label": compact ? "Expand the opponents" : "Shrink the opponents",
      onclick: () => { compact = !compact; store("lanCompact", compact); render(); } },
    h("span", { html: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="${compact ? "M6 9l6 6 6-6" : "M6 15l6-6 6 6"}"/></svg>` }));
    map.replaceChildren(toggle, ...players.filter((p) => p !== me).map((p) => {
      const on = p.id === shownId, taken = (me.commander_damage_by || {})[p.id] || 0;
      const details = [`${p.name}: ${p.life} life`, `hand ${p.hand}`, `library ${p.library}`, `graveyard ${p.graveyard.length}`];
      if (p.poison) details.push(`poison ${p.poison}/10`);
      if (taken) details.push(`commander damage to you ${taken}/21`);
      const q = h("button", {
        class: "q" + (on ? " on" : "") + (p.lost ? " out" : "") + (s.active_id === p.id ? " turn" : "") + (wanted.has(p.id) && !on ? " can" : ""),
        role: "tab", "aria-selected": on,
        title: details.join(" · ") + (on ? "" : ` — click to see ${p.name}'s board`),
        onclick: () => show(p.id),
      },
      h("div", { class: "qn" }, p.name),
      h("div", { class: "ql" }, h("b", null, p.life)),
      h("div", { class: "qb" },
        s.active_id === p.id ? h("i", { class: "turn", title: "Their turn" }) : null,
        s.priority_id === p.id ? h("i", { class: "prio", title: "Has priority" }) : null,
        attacking.has(p.id) ? h("i", { class: "atk", title: "Attacking you", html: SWORDS }) : null,
        p.won ? h("span", null, "Won") : p.lost ? h("span", null, "Out") : null));
      artInto(q, commanderOf(p));
      q.dataset.ids = p.id;  // the stack's links point here when the player's board is not in view
      return q;
    }));
  }

  // ---------- who attacks whom ----------
  // In front of each attacking creature (below it on the opponent's half, above it on yours) a
  // label shows whom it attacks: that player's commander and name, or the planeswalker. While
  // you declare attackers, the creatures you picked show where they are going.
  // The labels live in a layer over the page, placed on the cards once the table is laid out
  // (as the stack's lines are): the fields scroll, and would cut off a label outside its card.
  let tagTimer = null;
  function renderAttackTags(v) {
    let layer = $("attackTags");
    if (!layer) { layer = h("div", { id: "attackTags" }); document.body.append(layer); }
    layer.replaceChildren();
    if (!v.state || !v.state.everyone) return;
    const targets = new Map();
    for (const g of v.state.combat || []) for (const id of g.attackers) targets.set(id, g.defender);
    if (plan && plan.kind === "attackers") for (const [id, list] of plan.picks) targets.set(id, list[0]);
    for (const [id, defender] of targets) {
      const tag = attackTag(v.state, defender);
      tag.dataset.card = id;
      layer.append(tag);
    }
    requestAnimationFrame(placeTags);
    clearTimeout(tagTimer); tagTimer = setTimeout(placeTags, 300);
  }
  function placeTags() {
    const seen = new Set();
    for (const tag of document.querySelectorAll("#attackTags .atk-tag")) {
      const card = document.querySelector(`#table .card[data-ids~="${CSS.escape(tag.dataset.card)}"]`);
      tag.hidden = !card || seen.has(card);  // a stack of identical tokens shows one label
      if (tag.hidden) continue;
      seen.add(card);
      const r = card.getBoundingClientRect(), below = !!card.closest("#oppField");
      tag.classList.toggle("below", below);
      tag.style.left = r.left + r.width / 2 + "px";
      tag.style.top = (below ? r.bottom + 4 : r.top - 4) + "px";
      // Never wider than its card (the name is cut short), text sized to the card.
      tag.style.maxWidth = r.width + "px";
      tag.style.fontSize = Math.max(8.5, Math.min(11.5, r.width * 0.12)).toFixed(1) + "px";
    }
  }
  document.addEventListener("scroll", () => requestAnimationFrame(placeTags), true);
  addEventListener("resize", () => requestAnimationFrame(placeTags));
  function attackTag(s, defender) {
    const player = s.everyone.find((p) => p.id === defender);
    const walker = player ? null : ((raw && raw.state) || s).battlefield.find((c) => c.id === defender);
    const mine = player && player.side === "you";
    const name = player ? (mine ? "You" : player.name) : walker ? walker.name : "?";
    const face = h("i");
    artInto(face, player ? commanderOf(player) : walker && walker.name);
    return h("div", { class: "atk-tag" + (mine ? " you" : ""), title: `Attacking ${name}` },
      face, h("span", null, "→ " + name));
  }

  const controllerOf = (id) => (raw.state.battlefield.find((c) => c.id === id) || {}).controller_id;
  const commanderOf = (p) => (p.command[0] && p.command[0].name) || (p.deck && p.deck.commander);
  // The stack's lines to an object of an opponent out of view (a target, a source) end at
  // that player's square in the seat column.
  function anchorFor(place) {
    const s = view && view.state;
    if (!s || !s.everyone || !place.owner_id || s.players.some((p) => p.id === place.owner_id)) return null;
    return document.querySelector(`#seatMap [data-ids~="${CSS.escape(place.owner_id)}"]`);
  }
  const cornerWidth = () => { const map = $("seatMap"); return map ? map.offsetWidth + 10 : 0; };

  // ---------- whom to attack ----------
  // Clicking a creature that can attack several players or planeswalkers fans their cards out
  // above it, sized to that creature's card: commander (or planeswalker), name, life (or
  // loyalty), the damage the creatures already sent there would deal, and how many untapped
  // creatures could block. Click one to attack it; click elsewhere or Esc to keep the creature home.
  let fan = null;
  function chooseDefender(options, ev, choose) {
    closeFan();
    const whole = raw.state;
    const blockers = (id) => whole.battlefield.filter((c) => c.controller_id === id && c.creature && !c.tapped && !c.phased_out).length;
    const damage = plannedDamage(options[0].source);
    // In turn order, each player's planeswalkers right after them.
    const seat = (o) => {
      const at = whole.players.findIndex((p) => p.id === o.target);
      return at >= 0 ? at * 2 : whole.players.findIndex((p) => p.id === controllerOf(o.target)) * 2 + 1;
    };
    const cards = [...options].sort((a, b) => seat(a) - seat(b)).map((o) => {
      const player = whole.players.find((p) => p.id === o.target);
      const walker = player ? null : whole.battlefield.find((c) => c.id === o.target);
      const owner = player || whole.players.find((p) => walker && p.id === walker.controller_id) || {};
      const face = h("div", { class: "fa" });
      artInto(face, player ? commanderOf({ ...player, deck: deckOf(raw, player.name) }) : walker && walker.name);
      const n = blockers(owner.id), dealt = damage.get(o.target) || 0;
      return h("button", { class: "fan-card" + (player ? "" : " walker"), onclick: (e) => { e.stopPropagation(); closeFan(); choose(o); } },
        face,
        h("b", null, player ? player.name : walker ? walker.name : o.label),
        player ? h("div", { class: "fl" }, h("strong", null, player.life), " life")
          : h("div", { class: "fl" }, h("strong", null, (walker && walker.counters && walker.counters.loyalty) ?? "?"), " loyalty"),
        player ? null : h("small", null, `${owner.name || "?"}'s planeswalker`),
        h("div", { class: "fd" + (dealt ? "" : " none"), title: "Total power of the creatures you already sent here" },
          h("strong", null, dealt), " damage"),
        h("div", { class: "fb" + (n ? "" : " none") }, n === 1 ? "1 blocker" : `${n} blockers`));
    });
    const mid = (cards.length - 1) / 2;
    cards.forEach((c, i) => {
      c.style.setProperty("--rot", `${((i - mid) * 8).toFixed(1)}deg`);
      c.style.setProperty("--lift", `${(Math.abs(i - mid) * 10).toFixed(0)}px`);
      c.style.animationDelay = `${i * 35}ms`;
    });
    const anchor = (ev.target.closest && ev.target.closest(".card")) || ev.target;
    const r = anchor.getBoundingClientRect();
    fan = h("div", { class: "fan", role: "menu", "aria-label": "Attack whom" }, cards);
    // Each card a little wider than the attacking creature's card; text and art scale with it.
    fan.style.setProperty("--fw", Math.max(72, Math.min(130, r.width * 1.15)).toFixed(0) + "px");
    document.body.append(fan);
    const w = fan.offsetWidth, hgt = fan.offsetHeight;
    fan.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
    fan.style.top = Math.max(8, r.top - hgt - 14) + "px";
  }
  // Damage the creatures already chosen to attack would deal to each defender (their power),
  // without `attacker`, the creature being asked about.
  function plannedDamage(attacker) {
    const dealt = new Map();
    if (!plan || plan.kind !== "attackers") return dealt;
    for (const [id, list] of plan.picks) {
      if (id === attacker || !list.length) continue;
      const card = raw.state.battlefield.find((c) => c.id === id);
      dealt.set(list[0], (dealt.get(list[0]) || 0) + Math.max(0, (card && card.power) || 0));
    }
    return dealt;
  }
  function closeFan() { if (fan) { fan.remove(); fan = null; } }
  document.addEventListener("pointerdown", (e) => { if (fan && !fan.contains(e.target)) closeFan(); }, true);
  document.addEventListener("keydown", (e) => { if (fan && e.key === "Escape") { e.stopPropagation(); closeFan(); } }, true);

  // ---------- the lobby ----------
  // The lobby follows the table while it is open. Opened between games, it gives way to the next
  // game when someone starts it; opened during a game (New game), it stays.
  let lobbyBox = null, lobbyGame = null;
  async function openLobby() {
    lobbyOpen = true;
    lobbyGame = view ? view.game : null;
    if (!decks.length) { try { decks = (await (await fetch("/api/decks")).json()).decks; } catch (e) { decks = []; } }
    const name = h("input", { class: "lan-name", maxlength: 24, placeholder: "Your name", value: load("lanName", ""), "aria-label": "Your name" });
    const join = async (seat) => {
      store("lanName", name.value.trim());
      try { await post("/api/lobby/join", { name: name.value, seat }); reconnect(); } catch (e) { toast(e.message); }
    };
    name.addEventListener("keydown", (e) => { if (e.key === "Enter") join(null); });
    lobbyBox = h("div", { class: "lan-seats" });
    lobbyBox.join = join;
    dialog(h("div", { class: "lobby lan-lobby" },
      h("div", { class: "dh" }, h("div", null, h("h2", null, "A table among friends"),
        h("p", { class: "tag" }, `Friends on this network open ${shareUrl || location.origin + "/"} in their browser. Empty seats can go to XMage's AI.`)),
        h("button", { class: "btn ghost icon", onclick: () => { lobbyBox = null; closeDialog(); }, title: "Close" }, "✕")),
      h("div", { class: "db" },
        h("div", { class: "sect lan-join" }, h("h3", null, "Your name"), h("div", { class: "row" }, name,
          h("button", { class: "btn gold", onclick: () => join(null) }, "Take a seat"))),
        lobbyBox)));
    refreshLobby(view);
  }
  function refreshLobby(v) {
    if (!lobbyBox || !lobbyBox.isConnected || !v || !v.lobby) return;
    const L = v.lobby, mine = L.you, seated = mine != null;
    const act = (path, body) => post(path, body).catch((e) => toast(e.message));
    const deckPicker = (seat) => {
      const select = h("select", { class: "lan-deck", "aria-label": `Deck for ${seat.name}`, disabled: L.locked,
        onchange: (e) => act("/api/lobby/deck", { seat: seat.index, deck: e.target.value }) },
      decks.map((d) => h("option", { value: d.name, selected: d.name === seat.deck }, `${cap(d.name)} · ${d.commander || d.cards + " cards"}`)));
      return select;
    };
    const cards = L.seats.map((seat) => {
      const you = seat.index === mine;
      const body = [];
      const art = h("div", { class: "lan-art" });
      const deck = decks.find((d) => d.name === seat.deck);
      artInto(art, deck && deck.commander);
      if (seat.kind === "person") {
        body.push(h("b", null, seat.name + (you ? " (you)" : "")));
        body.push(you ? deckPicker(seat) : h("span", { class: "muted" }, deck ? cap(deck.name) : "No deck yet"));
        if (you && !L.locked) body.push(h("button", { class: "btn ghost", onclick: () => act("/api/lobby/leave").then(reconnect) }, "Leave the seat"));
      } else if (seat.kind === "bot") {
        body.push(h("b", null, seat.name), h("span", { class: "muted" }, "XMage's AI"));
        if (seated && !L.locked) {
          body.push(deckPicker(seat));
          body.push(h("label", { class: "lan-skill" }, "Skill",
            h("select", { onchange: (e) => act("/api/lobby/seat", { seat: seat.index, kind: "bot", skill: Number(e.target.value) }) },
              [[1, "1 · quick"], [3, "3"], [4, "4"], [6, "6 · strong, slow"]].map(([k, label]) => h("option", { value: k, selected: k === seat.skill }, label)))));
        }
      } else {
        body.push(h("b", null, seat.kind === "open" ? "Open seat" : "Closed"), h("span", { class: "muted" }, seat.kind === "open" ? "Waiting for a friend" : "Nobody plays here"));
      }
      const buttons = [];
      if (!L.locked && seat.kind !== "person") {
        if (!you) buttons.push(h("button", { class: "btn", onclick: () => lobbyBox.join(seat.index) }, seated ? "Move here" : "Sit here"));
        if (seated && seat.kind !== "bot") buttons.push(h("button", { class: "btn", onclick: () => act("/api/lobby/seat", { seat: seat.index, kind: "bot" }) }, "Add a bot"));
        if (seated && seat.kind !== "closed") buttons.push(h("button", { class: "btn ghost", onclick: () => act("/api/lobby/seat", { seat: seat.index, kind: "closed" }) }, "Close"));
        if (seated && seat.kind === "closed") buttons.push(h("button", { class: "btn ghost", onclick: () => act("/api/lobby/seat", { seat: seat.index, kind: "open" }) }, "Open"));
      }
      return h("div", { class: `lan-seat ${seat.kind}` + (you ? " you" : "") }, art, h("div", { class: "lan-who" }, body), h("div", { class: "row" }, buttons));
    });
    const playing = L.seats.filter((x) => x.kind === "person" || x.kind === "bot");
    const first = h("div", null, h("h3", null, "Who goes first"),
      h("div", { class: "seg" }, [[null, "Random"], ...playing.map((x) => [x.index, x.name])].map(([k, label]) =>
        h("button", { class: L.first === k ? "on" : "", disabled: !seated || L.locked, onclick: () => act("/api/lobby/first", { seat: k }) }, label))));
    const format = { "duel": "Duel", "commander-duel": "Commander duel", "free-for-all": "Free-for-all", "commander-free-for-all": "Commander free-for-all" }[L.format];
    const foot = L.locked
      ? h("div", { class: "foot" }, h("p", null, "A game is in progress. It can be abandoned for everyone; the seats change when it ends."),
        seated ? h("button", { class: "btn danger", onclick: () => act("/api/abandon") }, "Abandon the game") : null)
      : h("div", { class: "foot" }, h("p", null, L.problems.length ? L.problems.join(" ") : `${format || "Game"} for ${playing.length} players. Anyone at the table can start.`),
        h("button", { class: "btn gold", disabled: !seated || L.problems.length > 0 || v.status === "starting", onclick: start }, "Start the game"));
    document.querySelector(".lan-join").style.display = seated ? "none" : "";
    lobbyBox.replaceChildren(h("div", { class: "sect" }, h("h3", null, "Seats"), h("div", { class: "lan-grid" }, cards)),
      h("div", { class: "sect opts-row" }, first), foot);
  }
  async function start() {
    passUntil = false; passTurn = null; plan = null; arming = null; lastTurnKey = null; lastLife = {}; logTail = null;
    focus.manual = null;
    try { await post("/api/new"); shown = null; } catch (e) { toast(e.message); }
  }

  // ---------- going back to the start of a turn ----------
  // Anyone still playing may ask; every other person still playing must agree (bots always do).
  // The engine then plays the turn again from its start (XMage's own rollback).
  const REWIND_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 010 11H11"/></svg>';
  let rewindSeen = null;  // the request whose outcome was last announced
  function renderRewind(v) {
    let button = $("rewind");
    if (!button) {
      button = h("button", { class: "btn", id: "rewind", title: "Go back to the start of a turn (everyone at the table must agree)", onclick: askRewind },
        h("span", { html: REWIND_ICON }), h("span", null, "Rewind"));
      $("concede").before(button);
    }
    const s = raw && raw.state, me = s && s.players.find((p) => p.side === "you"), r = v.rewind;
    const playing = v.status === "playing" && !!s && s.turn > 0 && me && !me.lost && !v.result;
    button.hidden = !playing;
    button.disabled = !!(r && (r.status === "asking" || r.status === "approved"));
    let bar = $("rewindBar");
    if (!bar) { bar = h("div", { id: "rewindBar", class: "rewind-bar", role: "status" }); document.body.append(bar); }
    const key = r && `${r.by}:${r.turn}`;
    if (r && r.status === "refused") {
      if (rewindSeen !== key + ":refused") {
        rewindSeen = key + ":refused";
        toast(r.refused_by === r.by ? `${r.by} withdrew the request to go back.` : `${r.refused_by} does not want to go back to turn ${r.turn}.`);
      }
    }
    if (!r || r.status === "refused" || !playing) { bar.classList.remove("show"); return; }
    const answer = (accept) => post("/api/rewind/answer", { accept }).catch((e) => toast(e.message));
    const kids = [h("span", { class: "ri", html: REWIND_ICON })];
    if (r.status === "approved") kids.push(h("span", null, `Going back to the start of turn ${r.turn}…`));
    else if (r.your_vote) {
      kids.push(h("span", null, h("b", null, r.by), ` asks to go back to the start of turn ${r.turn}.`),
        h("button", { class: "btn gold", onclick: () => answer(true) }, "Accept"),
        h("button", { class: "btn ghost", onclick: () => answer(false) }, "Refuse"));
    } else {
      kids.push(h("span", null, `Going back to turn ${r.turn} once ${r.waiting.join(", ")} ${r.waiting.length > 1 ? "agree" : "agrees"}…`));
      if (r.mine) kids.push(h("button", { class: "btn ghost", onclick: () => answer(false) }, "Withdraw"));
    }
    bar.replaceChildren(...kids);
    bar.classList.add("show");
  }
  function askRewind() {
    const s = raw && raw.state;
    if (!s) return;
    const go = (turns) => { closeDialog(); post("/api/rewind", { turns }).catch((e) => toast(e.message)); };
    const choice = (turns, title, text) => h("button", { class: "rw-choice", onclick: () => go(turns) }, h("b", null, title), h("span", null, text));
    dialog(h("div", { class: "lobby rewind-ask" },
      h("div", { class: "dh" }, h("div", null, h("h2", null, "Go back"),
        h("p", { class: "tag" }, "The game is played again from the start of the turn you choose. Everyone still playing must agree; the bots always do.")),
        h("button", { class: "btn ghost icon", onclick: closeDialog, title: "Close" }, "✕")),
      h("div", { class: "db" },
        choice(0, `Start of this turn (${s.turn})`, `${possessive(activeName(s))} turn, from its untap step`),
        s.turn > 1 ? choice(1, `Start of the previous turn (${s.turn - 1})`, "From its untap step") : null)));
  }

  function afterRender(v) {
    renderSeatMap(v);
    renderRewind(v);
    renderAttackTags(v);
    if (fan && !(v.decision && v.decision.kind === "attackers")) closeFan();
    if (v.lobby) {
      const L = v.lobby, n = L.seats.filter((x) => x.kind === "person" || x.kind === "bot").length;
      if (!v.state) $("matchup").replaceChildren(h("span", { class: "vs" }, `Table · ${n} players`));
      else $("matchup").replaceChildren(h("span", { class: "vs" }, (v.state.everyone || v.state.players).map((p) => p.name).join(" · ")));
    }
    if (lobbyBox && lobbyBox.isConnected && v.status === "playing" && v.game !== lobbyGame) { lobbyBox = null; closeDialog(); }
    else refreshLobby(v);
  }

  // This script runs before the page's own (h() is not defined yet).
  const style = document.createElement("style");
  style.textContent = `
.seat-map { position: absolute; top: 12px; right: 14px; z-index: 15; --sq: clamp(96px, 7.6vw, 124px);
  display: flex; flex-direction: column; gap: 6px; padding: 5px; border-radius: 16px;
  background: rgba(0,0,0,.45); border: 1px solid var(--line-2); box-shadow: var(--shadow); backdrop-filter: blur(6px); }
.seat-map .q { position: relative; width: var(--sq); height: var(--sq); border-radius: 11px; overflow: hidden; border: 2px solid transparent; padding: 0;
  background: #1b2229 center 20%/cover; color: #fff; text-align: left; font: inherit; transition: border-color .15s, filter .15s, transform .12s; }
.seat-map .q::after { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,0,0,.15) 30%, rgba(0,0,0,.85)); }
.seat-map .q > * { position: relative; z-index: 1; }
.seat-map button.q { cursor: pointer; filter: saturate(.8) brightness(.85); }
.seat-map button.q:hover { filter: none; border-color: rgba(255,111,97,.55); }
.seat-map .q.on { filter: none; border-color: var(--bot); box-shadow: inset 0 0 0 1px rgba(255,111,97,.4); }
.seat-map .q.turn { box-shadow: 0 0 0 2px var(--gold) inset; }
.seat-map .q.can { border-color: var(--gold); animation: seatglow 2.2s ease-in-out infinite; }
.seat-map .q.pinged { border-color: #c77dff; box-shadow: 0 0 0 2px #c77dff, 0 0 18px rgba(199,125,255,.6); filter: none; }
.sm-toggle { height: 18px; display: grid; place-items: center; border: 0; border-radius: 7px; background: rgba(255,255,255,.06); color: var(--text-2); cursor: pointer; font-size: 10px; line-height: 1; padding: 0; }
.sm-toggle:hover { background: rgba(255,255,255,.14); color: var(--text); }
.seat-map.compact { gap: 4px; padding: 4px; }
.seat-map.compact .q { width: 148px; height: 30px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 9px;
  background: var(--glass-2) !important; filter: none; }
.seat-map.compact .q::after, .seat-map.compact .qb i.turn, .seat-map.compact .qb span { display: none; }
.seat-map.compact .qb { position: static; order: 3; gap: 3px; }
.seat-map.compact .qb i.prio { width: 7px; height: 7px; }
.seat-map.compact .qb i.atk { width: 16px; height: 16px; }
.seat-map.compact .q.out .qn { text-decoration: line-through; }
.seat-map.compact .qn, .seat-map.compact .ql { position: static; font-size: 12.5px; text-shadow: none; }
.seat-map.compact .qn { min-width: 0; flex: 1; } .seat-map.compact .ql b { font-size: 15px; text-shadow: none; }
.seat-map.compact .q.turn { box-shadow: inset 3px 0 0 var(--gold); }
.seat-map .q.out { filter: grayscale(1) brightness(.5); }
.seat-map .qn { position: absolute; left: 8px; right: 8px; bottom: 6px; font-weight: 700; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-shadow: 0 1px 3px #000; }
.seat-map .ql { position: absolute; left: 8px; bottom: 22px; } .seat-map .ql b { font-size: 24px; font-variant-numeric: tabular-nums; text-shadow: 0 2px 6px #000; }
.seat-map .qb { position: absolute; top: 6px; right: 6px; display: flex; gap: 4px; align-items: center; }
.seat-map .qb i { width: 10px; height: 10px; border-radius: 50%; font-style: normal; }
.seat-map .qb i.turn { background: var(--gold); } .seat-map .qb i.prio { background: #fff; box-shadow: 0 0 6px #fff; }
.seat-map .qb i.atk { width: 20px; height: 20px; display: grid; place-items: center; color: #fff; background: var(--bad);
  box-shadow: 0 0 0 2px rgba(0,0,0,.55), 0 0 10px var(--bad); }
.seat-map .qb span { font-size: 10px; font-weight: 800; text-transform: uppercase; background: rgba(0,0,0,.6); padding: 2px 5px; border-radius: 5px; }
#attackTags { position: fixed; inset: 0; z-index: 9; pointer-events: none; }
.atk-tag { position: fixed; transform: translate(-50%, -100%); pointer-events: none; overflow: hidden;
  display: flex; align-items: center; gap: .4em; padding: .15em .7em .15em .15em; border-radius: 999px; white-space: nowrap;
  background: rgba(24, 10, 10, .92); border: 1px solid var(--bad); box-shadow: 0 4px 12px rgba(0,0,0,.55);
  font-size: 11px; font-weight: 700; color: #fff; animation: tagin .2s ease-out; }
@keyframes tagin { from { opacity: 0; } }
.atk-tag.below { transform: translate(-50%, 0); }
.atk-tag.you { border-color: var(--you); background: rgba(10, 18, 30, .92); }
.atk-tag span { overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.atk-tag i { width: 1.9em; height: 1.9em; border-radius: 50%; background: #333 center 20%/cover; border: 1.5px solid currentColor; flex: none; }
.fan { position: fixed; z-index: 60; display: flex; align-items: flex-end; gap: calc(var(--fw) * .05); padding: 10px 14px 0; pointer-events: none; }
.fan-card { pointer-events: auto; width: var(--fw); font-size: calc(var(--fw) * .095); display: flex; flex-direction: column; align-items: center; gap: .35em;
  padding: .9em .6em .7em; border-radius: 1.1em; border: 1px solid var(--line-2); background: linear-gradient(180deg, #1d2631, #10161d); box-shadow: 0 10px 24px rgba(0,0,0,.6);
  cursor: pointer; color: var(--text); font-family: inherit; text-align: center; line-height: 1.2; transform-origin: 50% 140%;
  transform: rotate(var(--rot)) translateY(var(--lift)); animation: fanin .22s ease-out backwards; transition: transform .12s, border-color .12s; }
.fan-card:hover { border-color: var(--bot); transform: rotate(var(--rot)) translateY(calc(var(--lift) - 10px)) scale(1.06); z-index: 2; }
.fan-card { position: relative; } .fan-card::after { content: ""; position: absolute; left: 0; right: 0; top: 100%; height: 0; }
.fan-card:hover::after { height: 14px; }
.fan-card .fa { width: 4.6em; height: 4.6em; border-radius: 50%; border: .22em solid var(--bot); background: #222 center 20%/cover; box-shadow: 0 4px 12px rgba(0,0,0,.5); flex: none; }
.fan-card.walker .fa { border-color: var(--gold); }
.fan-card b { font-size: 1.1em; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fan-card .fl { color: var(--text-3); } .fan-card .fl strong { font-size: 1.5em; color: var(--text); font-variant-numeric: tabular-nums; }
.fan-card small { font-size: .85em; color: var(--text-3); max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fan-card .fd, .fan-card .fb { width: 100%; font-weight: 700; padding: .25em .3em; border-radius: .45em; white-space: nowrap; }
.fan-card .fd { color: #ffb3aa; background: rgba(255,107,107,.16); } .fan-card .fd strong { font-size: 1.25em; color: #fff; }
.fan-card .fd.none { color: var(--text-3); background: rgba(255,255,255,.05); } .fan-card .fd.none strong { color: var(--text-2); }
.fan-card .fb { color: var(--warn); background: rgba(255,181,71,.1); } .fan-card .fb.none { color: var(--good); background: rgba(79,209,139,.1); }
@keyframes fanin { from { opacity: 0; transform: rotate(0deg) translateY(30px) scale(.8); } }
.lan-lobby .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.lan-name, .lan-deck, .lan-skill select { background: rgba(0,0,0,.35); color: var(--text); border: 1px solid var(--line-2); border-radius: 10px; padding: 9px 12px; font: inherit; }
.lan-name { min-width: 240px; } .lan-deck { max-width: 100%; }
.lan-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.lan-seat { position: relative; display: flex; flex-direction: column; gap: 10px; padding: 12px; border-radius: 14px; border: 1px solid var(--line); background: var(--glass); min-height: 190px; overflow: hidden; }
.lan-seat.you { border-color: var(--you); box-shadow: 0 0 0 3px rgba(78,168,255,.15); }
.lan-seat.open { border-style: dashed; } .lan-seat.closed { opacity: .55; }
.lan-art { height: 70px; margin: -12px -12px 0; background: #1b2229 center 22%/cover; filter: saturate(.8) brightness(.75); }
.lan-who { display: flex; flex-direction: column; gap: 6px; flex: 1; } .lan-who b { font-size: 15px; } .lan-who .muted { font-size: 12px; color: var(--text-3); }
.lan-skill { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-3); }
.rewind-bar { position: fixed; left: 50%; top: 66px; z-index: 70; transform: translate(-50%, -12px); opacity: 0; pointer-events: none;
  display: flex; align-items: center; gap: 12px; max-width: calc(100vw - 32px); padding: 10px 12px 10px 14px; border-radius: 14px;
  background: rgba(16, 22, 29, .96); border: 1px solid var(--gold); box-shadow: 0 10px 30px rgba(0,0,0,.6); font-size: 14px; color: var(--text);
  transition: opacity .2s, transform .2s; }
.rewind-bar.show { opacity: 1; transform: translate(-50%, 0); pointer-events: auto; }
.rewind-bar .ri { color: var(--gold-2); display: grid; place-items: center; }
.rewind-bar .btn { padding: 6px 12px; }
.rewind-ask .db { display: flex; flex-direction: column; gap: 10px; }
.rw-choice { display: flex; flex-direction: column; gap: 4px; align-items: flex-start; padding: 14px 16px; border-radius: 12px; cursor: pointer;
  border: 1px solid var(--line-2); background: var(--glass); color: var(--text); font: inherit; text-align: left; }
.rw-choice:hover { border-color: var(--gold); background: var(--glass-3); }
.rw-choice b { font-size: 15px; } .rw-choice span { font-size: 12.5px; color: var(--text-3); }
@media (max-width: 760px) { .seat-map { --sq: 72px; } #rewind span:last-child { display: none; } }
`;
  document.head.append(style);

  window.mzTable = { connect, lens, afterRender, openLobby, chooseDefender, anchorFor, cornerWidth };
})();
