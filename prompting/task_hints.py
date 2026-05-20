import numpy as np
from rocobench.envs import EnvState, MujocoSimEnv


def _get_obj(obs: EnvState, name: str):
    if hasattr(obs.objects, "get"):
        return obs.objects.get(name)
    return getattr(obs.objects, name, None)


def pack_hint(env: MujocoSimEnv, obs: EnvState) -> str:
    item_names = list(getattr(env, "item_names", []))
    slot_xposes = dict(getattr(env, "bin_slot_xposes", {}))
    if not item_names or not slot_xposes:
        return ""

    occupied = {}  # slot -> item
    for name in item_names:
        obj = _get_obj(obs, name)
        if obj is None:
            continue
        contacts = getattr(obj, "contacts", [])
        if "bin_inside" not in contacts:
            continue
        xpos = getattr(obj, "xpos", None)
        if xpos is None:
            continue
        nearest_slot = min(
            slot_xposes.items(),
            key=lambda kv: float(np.linalg.norm(np.array(xpos[:2]) - np.array(kv[1][:2])))
        )[0]
        occupied[nearest_slot] = name

    packed_list = ", ".join(f"{n} -> {s}" for s, n in occupied.items()) if occupied else "(none)"
    empty_slots = [s for s in slot_xposes.keys() if s not in occupied]
    empty_list = ", ".join(empty_slots) if empty_slots else "(none)"

    # ----- Per-round mutex assignment -----
    # Alice = ur5e_robotiq (front-row preferred). Bob = panda (back-row preferred).
    alice_state = getattr(obs, "ur5e_robotiq", None)
    bob_state = getattr(obs, "panda", None)

    def _inhand(state):
        if state is None:
            return None
        for c in getattr(state, "contacts", []) or []:
            if c in item_names and c not in occupied.values():
                return c
        return None

    alice_inhand = _inhand(alice_state)
    bob_inhand = _inhand(bob_state)

    def _row(slot):
        return "front" if "front" in slot else ("back" if "back" in slot else "mid")

    def _pop_slot(prefer_row, taken):
        ordered = sorted(
            (s for s in empty_slots if s not in taken),
            key=lambda s: (0 if _row(s) == prefer_row else (1 if _row(s) == "mid" else 2), s)
        )
        return ordered[0] if ordered else None

    def _ee_y(state):
        ee = getattr(state, "ee_xpos", None) if state is not None else None
        try:
            return float(ee[1])
        except Exception:
            return None

    held_in_round = set(filter(None, [alice_inhand, bob_inhand]))

    def _pick_candidate(prefer_robot, taken_items):
        # nearest unpacked, currently-on-table item to this robot's gripper y
        candidates = []
        for name in item_names:
            if name in occupied.values() or name in taken_items or name in held_in_round:
                continue
            o = _get_obj(obs, name)
            if o is None:
                continue
            xpos = getattr(o, "xpos", None)
            if xpos is None:
                continue
            if "bin_inside" in getattr(o, "contacts", []):
                continue
            candidates.append((name, float(xpos[1])))
        if not candidates:
            return None
        ee_y = _ee_y(alice_state if prefer_robot == "Alice" else bob_state)
        if ee_y is None:
            # fall back: Alice prefers small y (front), Bob prefers large y (back)
            return min(candidates, key=lambda kv: kv[1])[0] if prefer_robot == "Alice" \
                else max(candidates, key=lambda kv: kv[1])[0]
        return min(candidates, key=lambda kv: abs(kv[1] - ee_y))[0]

    taken_slots, taken_items = set(), set()
    alice_line, bob_line = "", ""

    if alice_inhand:
        slot = _pop_slot("front", taken_slots)
        if slot:
            taken_slots.add(slot)
            alice_line = f"Alice (holding {alice_inhand}): PLACE {alice_inhand} {slot}"
        else:
            alice_line = f"Alice (holding {alice_inhand}): no empty slot left, pick a different empty slot"
    if bob_inhand:
        slot = _pop_slot("back", taken_slots)
        if slot:
            taken_slots.add(slot)
            bob_line = f"Bob (holding {bob_inhand}): PLACE {bob_inhand} {slot}"
        else:
            bob_line = f"Bob (holding {bob_inhand}): no empty slot left, pick a different empty slot"

    if not alice_line:
        pick = _pick_candidate("Alice", taken_items)
        if pick:
            taken_items.add(pick)
            alice_line = f"Alice (empty gripper): PICK {pick}"
        else:
            alice_line = "Alice (empty gripper): nothing left to pick, WAIT"
    if not bob_line:
        pick = _pick_candidate("Bob", taken_items)
        if pick:
            taken_items.add(pick)
            bob_line = f"Bob (empty gripper): PICK {pick}"
        else:
            bob_line = "Bob (empty gripper): nothing left to pick, WAIT"

    return (
        "[Pack Hard Rules]\n"
        "- Reachability (from env): Alice (ur5e_robotiq) reaches y in [-0.4, 1.5]; Bob (panda) reaches y in [0, 1.5].\n"
        "- Bin row y~0.5. Front slots (y~0.41) prefer Alice; back slots (y~0.58) prefer Bob; both can reach middle slots.\n"
        "- PATH last point z MUST be >= object_z + 0.05. The controller will lower the gripper to grasp height; do not plan below the table.\n"
        f"- Already packed items (DO NOT PICK again): {packed_list}\n"
        f"- Empty slots (valid PLACE targets only): {empty_list}\n"
        "- [Slot Mutex] Alice and Bob MUST NOT target the same bin slot in one round. Never PLACE into an occupied slot.\n"
        "- [Row Split] Alice -> front-row slot, Bob -> back-row slot (use a middle slot only if its row is exhausted).\n"
        "- [Non-Crossing Corridor] To avoid mid-air path conflict during parallel PLACE: ALL of Alice's PATH waypoints MUST have y <= 0.45; ALL of Bob's PATH waypoints MUST have y >= 0.55. Pick approach/lift waypoints that respect this corridor so the two arms never cross.\n"
        "- [No WAIT] PackGroceryTask does NOT accept WAIT/MOVE; every round each robot MUST output PICK or PLACE.\n"
        "[Pack Round Assignment] (follow unless physically impossible):\n"
        f"  - {alice_line}\n"
        f"  - {bob_line}\n"
    )


def sandwich_hint(env: MujocoSimEnv, obs: EnvState) -> str:
    recipe_order = list(getattr(env, "recipe_order", []) or [])
    food_items = list(getattr(env, "food_items", []) or [])
    if not recipe_order or not food_items:
        return ""

    progress = []
    board = _get_obj(obs, "cutting_board")
    if board is not None and "bread_slice1" in getattr(board, "contacts", []):
        progress.append(recipe_order[0])
        for i in range(len(recipe_order) - 1):
            cur = _get_obj(obs, recipe_order[i])
            if cur is None:
                break
            if recipe_order[i + 1] in getattr(cur, "contacts", []):
                progress.append(recipe_order[i + 1])
            else:
                break

    next_item = recipe_order[len(progress)] if len(progress) < len(recipe_order) else None

    left_items, right_items = [], []
    for f in food_items:
        if f in progress:
            continue
        obj = _get_obj(obs, f)
        if obj is None:
            continue
        xpos = getattr(obj, "xpos", None)
        if xpos is None:
            continue
        (left_items if xpos[0] < 0 else right_items).append(f)

    if next_item is None:
        next_line = "Recipe complete."
        assignee_line = ""
    else:
        obj = _get_obj(obs, next_item)
        if obj is None or getattr(obj, "xpos", None) is None:
            next_line = f"NEXT item to place: {next_item} (side unknown - re-check scene)."
            assignee_line = ""
        else:
            side = "left" if obj.xpos[0] < 0 else "right"
            assignee = "Dave" if side == "left" else "Chad"
            next_line = f"NEXT item to place: {next_item}; currently on {side} side; ONLY {assignee} can PICK/PUT it."
            assignee_line = "- The OTHER robot must output WAIT this round unless it can pre-PICK a later recipe item it can reach.\n"

    return (
        "[Sandwich Hard Rules]\n"
        f"- Recipe (this episode): {', '.join(recipe_order)}\n"
        f"- Stack progress so far: {', '.join(progress) if progress else '(none)'}\n"
        f"- {next_line}\n"
        f"{assignee_line}"
        "- Side rule: Dave reaches x<0 (left), Chad reaches x>0 (right). Do NOT PICK an item on the wrong side.\n"
        f"- Remaining items on LEFT (Dave only): {', '.join(left_items) if left_items else '(none)'}\n"
        f"- Remaining items on RIGHT (Chad only): {', '.join(right_items) if right_items else '(none)'}\n"
        "- Hard ban: do NOT PICK an item already stacked; do NOT PICK from the wrong side.\n"
    )


def cabinet_hint(env: MujocoSimEnv, obs: EnvState) -> str:
    cabinet_pos = np.array(getattr(env, "cabinet_pos", np.zeros(3)))
    align_threshold = float(getattr(env, "align_threshold", 0.25))
    left_side = cabinet_pos[0] < 0

    def _object_state(name: str):
        obj = _get_obj(obs, name)
        xpos = None if obj is None else getattr(obj, "xpos", None)
        if xpos is None:
            return "unknown", None
        xpos = np.array(xpos)
        coaster = np.array(env.coaster_pos[f"{name}_coaster"])
        if np.linalg.norm(xpos - coaster) < align_threshold:
            return "coaster", xpos
        if np.linalg.norm(xpos - cabinet_pos) < 0.35:
            return "cabinet", xpos
        return "outside", xpos

    def _held_object(agent_name: str):
        robot_name = env.robot_name_map_inv[agent_name]
        state = getattr(obs, robot_name, None)
        if state is None:
            return None
        contacts = set(getattr(state, "contacts", []) or [])
        for name in ["cup", "mug"]:
            if name in contacts:
                return name
        return None

    door_desp = ""
    try:
        door_desp = env.describe_cabinet(obs, include_coords=False)
    except Exception:
        door_desp = ""

    lines = ["[Cabinet Hard Rules]"]
    if "left door is open" in door_desp and "right door is open" in door_desp:
        lines.append("- Both doors are open. Robots already holding door handles should WAIT to keep the cabinet open unless a stricter recovery rule below applies.")

    held_any = False
    for agent_name in ["Alice", "Bob", "Chad"]:
        held = _held_object(agent_name)
        if held is None:
            continue
        held_any = True
        lines.append(f"- {agent_name} is already holding {held}; do NOT assign another PICK to {agent_name}. Continue by placing {held} on {held}_coaster.")

    recovery_lines = []
    for name in ["cup", "mug"]:
        state, xpos = _object_state(name)
        if state == "coaster":
            recovery_lines.append(f"- {name} is already on its coaster; do NOT PICK it again.")
            continue
        if state != "outside" or xpos is None:
            continue
        recovery_lines.append(
            f"- Recovery priority: {name} is outside cabinet and not on its coaster at "
            f"({xpos[0]:.2f}, {xpos[1]:.2f}, {xpos[2]:.2f}); recover {name} before switching to another object."
        )
        if name == "cup" and (xpos[1] >= 1.0 or xpos[2] <= 0.0):
            recovery_lines.append("- Abnormal cup state: when cup.y >= 1.0 or cup.z <= 0.0, do NOT assign Chad to PICK cup PLACE cup_coaster.")
        if name == "cup" and left_side:
            recovery_lines.append("- Left-side cabinet reminder: be conservative with Chad on cup recovery; prefer Alice when Chad repeatedly causes unreachable or collision states.")

    if recovery_lines:
        lines.extend(recovery_lines)
    elif not held_any:
        lines.append("- No object is currently outside cabinet. Finish the remaining in-cabinet object pickups only after both doors are open and held open.")

    return "\n".join(lines) + "\n"


def build_task_hint(env: MujocoSimEnv, obs: EnvState) -> str:
    cls_name = env.__class__.__name__
    try:
        if cls_name == "PackGroceryTask":
            return pack_hint(env, obs)
        if cls_name == "CabinetTask":
            return cabinet_hint(env, obs)
        if cls_name == "MakeSandwichTask":
            return sandwich_hint(env, obs)
    except Exception as exc:
        print(f"[task-hint] skipped due to: {exc}")
    return ""
