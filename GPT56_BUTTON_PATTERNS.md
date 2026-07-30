# GPT-5.6 (Sol Pro) ONNX Download Button Patterns — harvest

Observed in the NeuroGolf v1 0710 project (ChatGPT 5.6 Sol Pro replies). ChatGPT recently
changed how generated files are surfaced, so a harvest must detect **several** button shapes. This doc is
the working reference for the harvest detector; update as new shapes appear.

## ⚠️ EXHAUST EVERY OPTION BEFORE MARKING A BUTTON STALE / INERT (read this first)

"Inert" / "dead pod" is the LAST resort, not the first guess. Most false-inert reports were really a coordinate
miss, a stale ref, an open preview panel, or a read fired too early. A beating candidate is worth minutes of
retrying — do NOT log a blocker until you have genuinely tried ALL of the following and each produced **zero**
new files in `~/Downloads`:

1. **Confirm the page actually rendered.** get_page_text empty / title still "ChatGPT" / a spinner = still
   loading. Wait 3–6s and re-read. NEVER click before the reply text + button are visible.
2. **Close any open file-preview side-panel first** (its ✕). An open panel narrows the left column and shifts
   every chip's coordinate — clicks then land in empty space. Close the "rate limit" popup message too if it exists.
3. **Use SCREENSHOT-space coordinates, not JS `getBoundingClientRect`.** Screenshot ≈1568px wide, viewport
   ≈1728px → raw rect coords land ~9% too far right and miss. Screenshot the reply, click the link/chip AS IT
   APPEARS in the screenshot. (This was the #1 cause of false "inert".)
4. **Try BOTH affordances.** The blue underlined "Download …" LINK **and** the grey "Saved filename:"/"Saved
   file:" CHIP are each a real one-click download — click whichever you didn't try. `find` often returns the
   CHIP's ref (clicking the ref does nothing); screenshot + coordinate-click the LINK instead.
5. **Target the LATEST turn's button.** `find`/`querySelectorAll` can return a STALE earlier-turn button whose
   pod expired; the current turn's visible link downloads fine.
6. **Click 2–3× with a longer wait (~9s)** and re-verify by mtime: `find ~/Downloads -maxdepth 1
   -name 'taskNNN*.onnx' -newermt '-30s'` (the mover's snapshot diff can miscount vs shared `~/Downloads`).
7. **Re-screenshot after ANY layout change** (panel open/close, reflow) and re-read coords — never reuse a coord.
8. Try all the ways to click buttons which is stated below.
9. Only AFTER all of the above yield nothing: it's a genuine dead pod OR a phantom affordance (see those
   sections below). For a real beating candidate, send ONE recovery probe on Extra High/Pro; for a marginal
   win, log a blocker and move on. Do NOT loop probes.

Rule of thumb: **the buttons are downloadable far more often than they are inert.** Assume operator error first.

## How the deliverable is surfaced

The reply ends with a metrics table (Score / Cost / Memory bytes / Params / Canonical pass, columns
`Previous | Current | Delta`) and then a download affordance. The affordance is a **plain `<button>`** (or
occasionally an `<a>`), NOT a blob link, NOT a file-attachment card. There are no `[download]` anchors and no
`blob:`/`onnx` hrefs — querying those finds nothing. So detection must be by the **button's visible text**.

## Pattern A — "Download the …" prefixed button (BEATING CANDIDATE)

The button text starts with "Download" and names a candidate. Harvest these (gate on cost < champion).
Wordings seen:

- `Download task048_m2_p11.onnx`  ← includes the literal `taskNNN_mX_pY.onnx` filename
- `Download task002_m372_p204.onnx`
- `Download task012_p3k5.onnx`  ← nickname filename (no m/p) — read Current cost from the table instead
- `Download the passing task029 ONNX candidate`  ← "ONNX" not ".onnx"
- `Download the verified task013 ONNX graph`
- `Download the validated ONNX artifact`
- `Download the verified m34/p133 ONNX graph`

## Pattern B — bare-filename chip button (BEATING CANDIDATE, NEW UI)

The reply prints a `**Download:**` label (a `<strong>`), and the file itself renders as a **button whose
entire text is just the filename** — the word "Download" is NOT in the button:

- button text = `task033_m0_p82.onnx`

Detect with `/\.onnx\b/i` on the button text (not only `/^download/i`).

## Pattern C — champion baseline button (SKIP — NOT an improvement)

When the model could not beat the champion it still offers the rebuilt champion. Cost == champion, so it must
be **skipped**. Wordings seen:

- `Download the verified champion`
- `Download the rebuilt champion`
- `Download the verified task006 champion`
- `Download the verified 110-cost champion`
- `champ.onnx`  ← bare-filename chip of the CHAMPION build (task015). Matches `/\.onnx/` so the detector
  catches it, but the cost gate saves you: its Current cost == champion. Treat any `champ*.onnx` /
  `champion*.onnx` bare chip as Pattern C.

### Pattern B beats Pattern C only by cost

Both B (beating candidate chip `taskNNN_mX_pY.onnx`) and C (`champ.onnx`) are bare `.onnx` chips. The ONLY
reliable discriminator is **cost < champion** (from the `mX_pY` in the filename or the table Current cost).
Always apply the cost gate before downloading. Recovered example: task023 chip `task023_m753_p136.onnx`
(cost 889 < champ 1023, 29/29) was initially missed by a `/^download/`-only selector — the chip-aware
selector + cost gate caught it.

Note: the metrics table may show an aspirational lower "Current" cost (e.g. task014 showed Current 350 vs
champ 380) even though the model only **exported the champion** (declined to export the lower design). So do
NOT gate on the table's Current cost alone when the only button is a champion/baseline button — a
`champion|baseline` button is always the unchanged champion.

## Pattern D — descriptive-label button (no "download", no ".onnx")

Some replies render the file as a button whose text is a **description with an embedded `mX pY` cost**, no
"Download" word and no `.onnx`. Seen (task055):

- `Validated original champion, m0 p226`  ← Pattern C (champion) → skip
- `Best lower-cost diagnostic candidate, m0 p218 — fails canonical`  ← beating cost 218<226 but FAILS
  canonical → route to `unsafe/`
- `Experimental m8/p125 candidate — not valid for submission` (task035) → cost 133<137, not valid → `unsafe/`

Detect with a cost-signature regex on button text: `/\bm\s*\d+\s*[\/ ]?\s*p\s*\d+/i`, plus keywords
`candidate|diagnostic|validated (onnx|submission|artifact)|passing .*model`. A button that says
`fails canonical` / `not valid` / `diagnostic` / `experimental` → the graph fails canonical → `unsafe/`
(still a cheaper submit, the Space judges). A clean one → `safe/`.

## Not-downloadable (env reset) — BLOCKER

Some replies computed a beating candidate but the sandbox reset before export: e.g. task054 —
`"environment reset, so I cannot provide a trustworthy download link"` (claimed m2624_p749 cost 3373 <
champ 4464). There is NO button. Under a no-nudging harvest, log a blocker row and move on (do not probe).

## Noise to exclude

- `Download apps`, `Download the ChatGPT app`, `Download for macOS/Windows/iOS/Android` — ChatGPT UI chrome.
- `.onnx` strings inside `<code>` blocks — these are the prompt template `taskNNN_m{memory}_p{params}.onnx`,
  not real files. Only count `<button>`/`<a>` elements, not `<code>`.

## Working detector (JS)

```js
const cands = [...document.querySelectorAll('button,a')].filter(b => {
  const t = (b.textContent||'').replace(/\s+/g,' ').trim();
  const isDl = /^download\b/i.test(t) || /\.onnx\b/i.test(t);   // Pattern A or B
  return isDl && !/champion|baseline|download apps|chatgpt app|download for /i.test(t); // exclude C + chrome
});
```

**Gate:** harvest only if claimed cost < champion cost, where claimed cost =
`m+p` from a `taskNNN_mX_pY.onnx` filename if present, else the table's **Current** `Cost` row
(`^Cost\t<prev>\t<cur>`), else Current `Params` + `Memory bytes`.

**Route:** `safe/` if the reply's canonical fraction is all-pass (e.g. `30/30`, `33/33`, "VERDICT: PASS");
otherwise `unsafe/`. A `champion|baseline`-only reply = no beating candidate → skip entirely.

## Comprehensive fallback scan (when a file is mentioned but no button matches)

If the reply text mentions a beating `taskNNN_mX_pY.onnx` but the detector finds no button, enumerate leaf
elements and buttons whose text contains `download` or `.onnx` (excluding `<code>`/`<script>`) to find the
actual chip — this is how Pattern B was discovered.

## 2026-07-10 UPDATE — download buttons are NOT inert; Chrome blocks AUTOMATED multiple downloads

Harvesting run 0710b: the first ~10 ONNX download buttons downloaded fine via CDP click
(computer left_click on ref or coordinate). After that, EVERY subsequent CDP click produced NO
file — including on a button the USER then downloaded MANUALLY seconds later (task119). So:

- The buttons are **downloadable** (not inert). Manual OS clicks work.
- **Chrome silently blocks programmatic/automated downloads** after several from the same site
  ("Allow automatic downloads / multiple downloads" permission not granted). CDP-dispatched clicks
  count as automated → blocked. A permission chip appears in the address bar the FIRST time.
- Symptom: move script reports "0 newly-appeared onnx"; `find ~/Downloads -newermt '-30s'` empty;
  no modal, button visibly highlights on click.

**FIX (one-time, user action):** in Chrome, allow chatgpt.com to download multiple/automatic files
(click "Allow" on the address-bar download permission chip, or Site settings → Automatic downloads →
Allow). After that, CDP clicks download again and ALL still-live links can be harvested — no nudge
needed. If a link's sandbox pod has genuinely expired, THEN send a recovery nudge on Extra High
asking to re-export the exact taskNNN_mX_pY.onnx.

**Also:** a `Saved filename: taskNNN_mX_pY.onnx` code chip is DISPLAY-ONLY (not a download); the real
affordance is the separate `Download …` link/button. Page needs ~1.5s to render after navigate before
the response text/button is queryable.

## 2026-07-10 CRITICAL FIX — `find` returns the "Saved file:" CHIP, not the "Download" LINK
The replies often render BOTH: a `Saved file: taskNNN_mX_pY.onnx` code chip (display-only, y~340) AND a
separate blue underlined `Download taskNNN_mX_pY.onnx` LINK (y~402). The `find` tool frequently matches the
CHIP (same filename text) and returns its ref; clicking that ref does NOTHING (no download). This caused
MANY false "dead pod" blockers. FIX: SCREENSHOT the reply, locate the blue underlined "Download …" link
(below the Saved-file chip / above the action-icons row), and COORDINATE-CLICK that link. Confirmed: task111
recovered this way after ref-click on the chip failed. Downloads themselves work fine (CDP block was
transient). Only genuinely expired pods (e.g. task096 original after ~3h) truly fail.

## 2026-07-11 CONFIRMED — descriptive `Download the ONNX file` button: COORDINATE-click fails, JS `.click()` WORKS

The Pattern-D-adjacent **`Download the ONNX file`** affordance (generic label, no filename, no ".onnx" in the
text) renders as a `<button class="behavior-btn ...">` — it triggers the file export through a **React
synthetic `onClick` handler**, NOT an `<a href>`/blob (`href` is `null`, `download` attr `null`, `disabled`
false, `cursor-pointer`). Confirmed on **task071** (v2, 2026-07-11):

- `computer` **left_click at the button's exact centered screenshot coordinate → NO download** (button
  visibly highlights, but the React handler does not fire from the CDP mouse event). Repeated coordinate
  clicks produced zero files. This looks EXACTLY like an "inert" button but is NOT.
- JS **`element.click()`** on the same button → **downloads immediately** (`task071_m176_p39.onnx` landed in
  `~/Downloads`). The user reproduced the same download manually, confirming the pod was live the whole time.

**RULE:** for a `behavior-btn` / descriptive `Download the ONNX file` button, do NOT coordinate-click and do
NOT conclude inert. Locate it and call `element.click()` in JS:
```js
const b=[...document.querySelectorAll('button')].find(e=>/Download the ONNX file/i.test(e.innerText)); b.click();
```
Then verify by mtime (`ls -la ~/Downloads/taskNNN*` or `find ~/Downloads -maxdepth 1 -name 'taskNNN*.onnx' -newermt '-30s'`).
Only after BOTH coordinate-click AND JS `.click()` yield zero files is a recovery probe justified.

(Prefixed `Download taskNNN_mX_pY.onnx` links and the `Download:` / `Saved file:` blue underlined LINKS still
download fine via screenshot-space coordinate-click — the JS-`.click()` path is specifically the fix for the
generic `behavior-btn`. When unsure which shape you have, try JS `.click()` on the matched element FIRST; it
works for all button shapes and sidesteps the coordinate-space pitfall entirely.)

## Switching the base model GPT-5.6 → GPT-5.5 in the model dropdown

The composer model pill shows the **effort tier** (Instant / Medium / High / Extra High / Pro), NOT the
base model. The base model (GPT-5.6 Sol vs GPT-5.5) lives in a **submenu** at the bottom of the picker.

Steps:
1. Click the **model pill** in the composer (right side of the input box, e.g. reads `Pro` / `Extra High`).
   This opens the model dropdown. (Keyboard shortcut alternative: **⌃⇧M** / Ctrl+Shift+M opens the picker.)
2. The **top section ("Intelligence")** lists EFFORT tiers only — `Instant 5.5`, `Medium`, `High`,
   `Extra High`, `Pro`. Clicking these changes effort, **not** the base model. Do NOT rely on them to switch
   to 5.5.
3. At the **BOTTOM of the dropdown** is the base-model row: **`GPT-5.6 Sol ›`** (with a right-chevron ›).
   Hover/click it to open the **base-model submenu**. (Its Y position drifts with window height — screenshot
   to locate, then coordinate-click the row.)
4. In the submenu, select **`GPT-5.5`** (the base model entry). The dropdown closes.
5. Re-open the pill if you also need to set the effort tier (e.g. `Pro` or `Extra High`) under GPT-5.5.
6. **VERIFY the pill:** it must now read **`5.5 Pro`** / **`5.5 Extra High`** (with the `5.5` prefix). A
   **bare `Pro` / `Extra High`** (no `5.5`) means the base model is STILL GPT-5.6 Sol — the switch did NOT
   take. Re-do steps 3–4.

⚠️ In these projects the base model **silently reverts to GPT-5.6 Sol on each navigation / new chat**,
so re-select GPT-5.5 (steps 3–4) and re-verify the `5.5` pill prefix **before every send** — a bare `Pro`
pill ≠ 5.5 Pro.

## Coordinate space: `computer` click coords are SCREENSHOT-space, NOT JS `getBoundingClientRect`

**The single most important download-click lesson.** The `computer` left_click tool expects coordinates in
the **screenshot's pixel space** (e.g. 1568 wide), but `document.querySelector(...).getBoundingClientRect()`
returns **CSS/viewport pixels** of the real window (often **1728** wide when the ChatGPT sidebar + wide
layout are active). They differ by a scale factor (**1568/1728 ≈ 0.907** horizontally; height scales too).

Symptom: you locate a download chip/link via JS, get e.g. `{x:778,y:335}`, click `(778,335)`, and **nothing
downloads** (the click lands ~70px too far right / too low, hitting empty space or the wrong element — often
it opens/toggles the file-preview side-panel instead). You then wrongly conclude the chip is "inert" or the
pod is dead. It is NOT inert — your coordinate is in the wrong space.

**FIX (either one):**
- **Preferred:** take a `screenshot`, visually find the `Saved file: taskNNN_mX_pY.onnx` chip (or the blue
  `Download …` link), and coordinate-click at the position **as it appears in the screenshot**. This is
  self-consistent — screenshot coords → click coords.
- Or scale JS rect coords: `click_x = round(rect_cx * screenshotWidth/window.innerWidth)`,
  `click_y = round(rect_cy * screenshotHeight/window.innerHeight)`.

**Confirmed (task269_m237_p0, 2026-07-11):** JS rect said chip at `(778,335)`; clicking it repeatedly only
toggled the md preview panel and never downloaded. Clicking the **screenshot** position `(400,447)` of the
same chip downloaded `task269_m237_p0.onnx` immediately. The onnx "Saved file:" chip **is** a real
one-click browser download — no preview panel, no separate Download button needed — as long as you click its
true screenshot coordinate with **no preview panel already open** (close the panel with its ✕ first; an open
panel narrows the left column and shifts every chip).

Corollary: after ANY panel open/close the left column reflows and every chip's Y moves — re-screenshot and
re-read coords before each click; never reuse a coord across a layout change.

## Download-button lessons — quick checklist (2026-07-11 harvest)

When a reply advertises a `taskNNN_mX_pY.onnx` beat, download it with this exact sequence:

1. **Close any open file-preview side-panel first** (click its ✕ at the panel's top-right). An open panel
   narrows the left conversation column and shifts every chip's coordinates — download clicks then miss.
2. **Screenshot** the reply. Visually find the download control:
   - a `Saved file: taskNNN_mX_pY.onnx` **grey code chip**, OR
   - a blue underlined `Download taskNNN_mX_pY.onnx` **link** (below the chip / above the action-icon row).
   Both are genuine one-click browser downloads.
3. **Coordinate-click at the position AS IT APPEARS IN THE SCREENSHOT** (screenshot pixel space ≈ 1568 wide).
   Do **not** click raw `getBoundingClientRect` coords (viewport ≈ 1728 wide) — they're ~9% too far right and
   the click misses (see the coordinate-space section above). This is the #1 cause of false "inert chip"/
   "dead pod" conclusions.
4. **Verify the file actually landed in `~/Downloads`** (`ls ~/Downloads | grep taskNNN`). "It didn't appear"
   = the click missed, NOT that the chip is inert. Re-screenshot (layout may have reflowed) and re-click.
5. **Move it to `safe/` immediately** after it appears (`~/Downloads` is shared / Chrome renames dupes to
   `(1)` — strip the `(1)` to the clean `taskNNN_mX_pY.onnx` name before the poller parses it).

Do NOT conclude a chip is inert until you've clicked its correct **screenshot-space** coordinate with no
preview panel open and confirmed nothing reached `~/Downloads`. The onnx "Saved file:" chip is downloadable.

## 2026-07-11 UPDATE — harvest: click the LATEST turn's link; `find` returns STALE refs

After the user re-enabled Chrome automatic downloads, ONNX download buttons work again. Hard-won lessons
from the 0711 harvest pass:

- **Both affordances download.** A `Download: taskNNN_mX_pY.onnx` blue link AND a bare `Saved filename:
  taskNNN_mX_pY.onnx` chip are BOTH clickable and both save the file (task330 downloaded via the bare
  "Saved filename" chip). Do NOT assume the "Saved filename" chip is display-only — click it aggressively.
- **The `find` tool / a JS `querySelectorAll` match can return a STALE earlier-turn button.** Clicking that
  ref downloads NOTHING (its sandbox pod expired). task339's `find` ref_327 clicked but produced 0 files;
  the SAME file downloaded fine when I screenshotted the LATEST turn and coordinate-clicked the visible blue
  `Download:` link. **Always target the latest turn's visible link** (screenshot → click the link you can
  see near the bottom of the reply, just above the action-icon row), not whatever ref `find` returns first.
- **Latest-turn-regressed = stale beat.** If a task's LATEST turn concluded "no lower-cost passing ONNX
  found / champion remains best" but an EARLIER turn had a beating chip (e.g. task329 cost78 in an older
  round), that earlier chip is inert — clicking it (coord, precise, or ref) all yield 0 files. The pod is
  gone. Either accept the loss or send a recovery nudge on Extra High to re-export the exact
  taskNNN_mX_pY.onnx.
- **Verify every download by mtime, not the mover's snapshot diff.** `touch /tmp/marker` before clicking,
  then `find ~/Downloads -maxdepth 1 -name 'taskNNN*.onnx' -newer /tmp/marker`. The harvest_move snapshot
  diff can miscount when a same-named file from a prior run already sits in Downloads.
- **Fast per-task triage (JS, one call):** enumerate `a,button` whose text matches
  `taskNNN_m\d+_p\d+\.onnx`. `n==0` → no-beat, skip. Else take the LAST match (latest turn),
  `scrollIntoView({block:'center'})`, read its `getBoundingClientRect` center, and coordinate-click that.
  Gate cost `m+p < champion` before moving to `safe/` (route `unsafe/` if the reply's VERDICT is FAIL).
- **A validated beat with NO button at all** (reply printed `VERDICT: PASS — submit it` for
  task338_m0_p198 but attached no link in that turn): the clickable button was in a slightly earlier turn
  (task338_m0_p198 button existed higher up) and downloaded fine via ref. So scan ALL clickable buttons for
  the task, pick the lowest-cost that actually downloads.

## 2026-07-11 — GENUINELY PHANTOM affordance: reply admits the file was never saved to the target name

The ONE case that IS truly inert, distinct from the coordinate-space miss and the stale-pod cases above.
Some GPT-5.6 replies render the download as a **plain non-link line** — `Download the validated ONNX
candidate` in ordinary body text, NOT a blue link, NOT a grey `.onnx` chip, no `<button>`/`<a>` — AND the
closing paragraph confesses the artifact was never materialized under the requested filename. Example
(task185): *"The validated artifact remained under its working name `modular.onnx` when execution ended,
rather than being physically copied to the requested filename; the downloadable file above is the exact
validated graph."*

- **Tell:** the "download" is plain prose (not the styled chip/blue link of Patterns A/B) **and** the text
  names a *different* working filename / says the file was "not physically copied" / points vaguely "above."
  No real chip exists → clicking the correct screenshot-space coordinate (verified twice) downloads nothing.
- **This is NOT a coordinate bug and NOT a re-clickable pod** — the file object was never exported. Re-clicking
  never helps; only a fresh re-export does.
- **Action:** for a MARGINAL win (task185 was only +0.11) log a blocker + skip — a recovery probe costs a
  send/cooldown slot not worth it. For a BIG win, send ONE recovery probe (`Give me the downloadable best ONNX
  file`) to force a real export, then harvest the fresh chip. Never burn multiple clicks on the phantom line.
