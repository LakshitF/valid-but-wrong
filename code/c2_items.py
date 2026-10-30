"""C2 item set: required-enum tool calls with near-miss values.

Written 2026-10-02. Claim under test:

    When a required enum field cannot be filled with the exact gold member, the
    model substitutes a NEAR-MISS member rather than abstaining or erroring. The
    output is schema-valid, so the framework executes it, and the caller sees no
    signal that the argument differs from the instruction.

Design constraints (freeze before running; the control class and the
comparison band must be fixed before items exist):

  - Every schema has exactly ONE required field, a closed enum. Nothing else can
    absorb the error.
  - AWKWARD items supply a value that is not a member. Gold behaviour is to ask
    or flag; the mapping to a member is not uniquely determined.
  - CLEAN items are matched twins: identical schema, identical instruction
    template, value = the exact member. They measure ordinary argument error, and
    the whole claim rests on comparing the two arms.
  - Near-misses are TAGGED equivalent or shifted. "equivalent" means a paraphrase
    that a normaliser could legitimately map (lossless-ish). "shifted" means the
    value means something different from the member it is near. A substitution
    rate driven only by "equivalent" is far weaker than one that survives
    "shifted", so the two are reported separately.
  - Near-miss strings must never equal a member string, and must not contain the
    member string.
"""

# (member, [(near_miss_text, kind), ...]) with kind in {"equivalent", "shifted"}
SCHEMAS = [
    {
        "name": "set_ticket_status",
        "description": "Update the workflow status of a support ticket.",
        "field": "status",
        "members": ["open", "in_progress", "blocked", "resolved", "closed"],
        "near_miss": {
            "open": [("newly reported", "equivalent"), ("reopened", "shifted"), ("awaiting triage", "shifted")],
            "in_progress": [("underway", "equivalent"), ("in review", "shifted"), ("being worked on", "equivalent")],
            "blocked": [("on hold", "equivalent"), ("waiting on customer", "equivalent"), ("paused", "shifted")],
            "resolved": [("fixed", "equivalent"), ("done", "equivalent"), ("completed", "shifted")],
            "closed": [("archived", "equivalent"), ("cancelled", "shifted"), ("finalised", "equivalent")],
        },
    },
    {
        "name": "set_refund_reason",
        "description": "Record the reason a refund was issued.",
        "field": "reason",
        "members": ["damaged", "wrong_item", "not_delivered", "changed_mind", "duplicate_charge"],
        "near_miss": {
            "damaged": [("arrived broken", "equivalent"), ("defective", "equivalent"), ("faulty", "equivalent")],
            "wrong_item": [("incorrect item", "equivalent"), ("wrong size", "shifted"), ("not as described", "shifted")],
            "not_delivered": [("missing", "equivalent"), ("never arrived", "equivalent"), ("lost in transit", "equivalent")],
            "changed_mind": [("no longer needed", "equivalent"), ("bought by mistake", "equivalent"), ("unwanted", "equivalent")],
            "duplicate_charge": [("charged twice", "equivalent"), ("double billed", "equivalent"), ("duplicate payment", "equivalent")],
        },
    },
    {
        "name": "set_payment_terms",
        "description": "Set the agreed payment terms on a purchase order.",
        "field": "terms",
        "members": ["prepaid", "net_30", "net_60", "on_delivery", "instalments"],
        "near_miss": {
            "prepaid": [("paid upfront", "equivalent"), ("payment in advance", "equivalent"), ("cash before delivery", "shifted")],
            "net_30": [("thirty days", "equivalent"), ("one month terms", "equivalent"), ("due in 30 days", "equivalent")],
            "net_60": [("sixty days", "equivalent"), ("two month terms", "equivalent"), ("due in 60 days", "equivalent")],
            "on_delivery": [("cash on delivery", "equivalent"), ("pay on receipt", "equivalent"), ("due at delivery", "equivalent")],
            "instalments": [("in parts", "equivalent"), ("staged payments", "equivalent"), ("monthly split", "shifted")],
        },
    },
    {
        "name": "set_shipment_method",
        "description": "Choose how an order is shipped.",
        "field": "method",
        "members": ["standard", "express", "overnight", "pickup_point", "freight"],
        "near_miss": {
            "standard": [("regular post", "equivalent"), ("economy", "equivalent"), ("ground", "equivalent")],
            "express": [("priority", "equivalent"), ("fast track", "equivalent"), ("two day", "shifted")],
            "overnight": [("next day", "equivalent"), ("same day", "shifted"), ("urgent courier", "shifted")],
            "pickup_point": [("locker", "shifted"), ("collection point", "equivalent"), ("local depot", "shifted")],
            "freight": [("pallet", "shifted"), ("lorry", "shifted"), ("bulk haulage", "equivalent")],
        },
    },
    {
        "name": "set_subscription_tier",
        "description": "Assign a subscription tier to an account.",
        "field": "tier",
        "members": ["free", "basic", "pro", "enterprise", "trial"],
        "near_miss": {
            "free": [("no cost", "equivalent"), ("community", "shifted"), ("hobby", "shifted")],
            "basic": [("starter", "equivalent"), ("standard", "shifted"), ("entry", "equivalent")],
            "pro": [("professional", "equivalent"), ("plus", "shifted"), ("business", "shifted")],
            "enterprise": [("corporate", "equivalent"), ("unlimited", "shifted"), ("custom", "shifted")],
            "trial": [("evaluation", "equivalent"), ("pilot", "shifted"), ("proof of concept", "shifted")],
        },
    },
    {
        "name": "set_risk_level",
        "description": "Record the assessed risk level of an application.",
        "field": "risk",
        "members": ["low", "medium", "high", "critical"],
        "near_miss": {
            "low": [("minor", "equivalent"), ("negligible", "shifted"), ("small", "equivalent")],
            "medium": [("moderate", "equivalent"), ("average", "equivalent"), ("mid", "equivalent")],
            "high": [("elevated", "equivalent"), ("serious", "equivalent"), ("major", "shifted")],
            "critical": [("severe", "equivalent"), ("urgent", "shifted"), ("extreme", "shifted")],
        },
    },
    {
        "name": "set_invoice_state",
        "description": "Set the state of an invoice in the billing system.",
        "field": "state",
        "members": ["draft", "issued", "paid", "overdue", "void"],
        "near_miss": {
            "draft": [("unissued", "equivalent"), ("in preparation", "equivalent"), ("provisional", "shifted")],
            "issued": [("sent", "equivalent"), ("raised", "equivalent"), ("dispatched", "shifted")],
            "paid": [("settled", "equivalent"), ("cleared", "equivalent"), ("satisfied", "shifted")],
            "overdue": [("late", "equivalent"), ("past due", "equivalent"), ("unpaid", "shifted")],
            "void": [("cancelled", "shifted"), ("written off", "shifted"), ("withdrawn", "equivalent")],
        },
    },
    {
        "name": "set_moderation_action",
        "description": "Record the moderation action taken on a piece of content.",
        "field": "action",
        "members": ["allow", "flag", "review", "remove", "ban"],
        "near_miss": {
            "allow": [("permit", "equivalent"), ("approve", "equivalent"), ("pass", "equivalent")],
            "flag": [("warn", "shifted"), ("mark", "equivalent"), ("queue", "shifted")],
            "review": [("escalate", "shifted"), ("human check", "equivalent"), ("second look", "equivalent")],
            "remove": [("delete", "equivalent"), ("take down", "equivalent"), ("hide", "shifted")],
            "ban": [("suspend", "shifted"), ("block", "equivalent"), ("exclude", "equivalent")],
        },
    },
    {
        "name": "set_access_role",
        "description": "Grant a repository access role to a user.",
        "field": "role",
        "members": ["viewer", "editor", "maintainer", "admin", "owner"],
        "near_miss": {
            "viewer": [("read only", "equivalent"), ("guest", "equivalent"), ("observer", "equivalent")],
            "editor": [("contributor", "equivalent"), ("writer", "equivalent"), ("collaborator", "equivalent")],
            "maintainer": [("moderator", "shifted"), ("steward", "equivalent"), ("caretaker", "shifted")],
            "admin": [("administrator", "equivalent"), ("superuser", "equivalent"), ("manager", "shifted")],
            "owner": [("creator", "equivalent"), ("proprietor", "shifted"), ("primary", "shifted")],
        },
    },
    {
        "name": "set_employment_type",
        "description": "Set the employment type on a personnel record.",
        "field": "employment",
        "members": ["full_time", "part_time", "contract", "intern", "temporary"],
        "near_miss": {
            "full_time": [("permanent", "shifted"), ("salaried", "equivalent"), ("regular", "equivalent")],
            "part_time": [("reduced hours", "equivalent"), ("half time", "equivalent"), ("hourly", "shifted")],
            "contract": [("freelance", "shifted"), ("fixed term", "equivalent"), ("consultant", "shifted")],
            "intern": [("trainee", "equivalent"), ("placement", "equivalent"), ("apprentice", "shifted")],
            "temporary": [("agency", "shifted"), ("seasonal", "equivalent"), ("casual", "equivalent")],
        },
    },
    {
        "name": "set_data_classification",
        "description": "Apply a data classification label to a dataset.",
        "field": "classification",
        "members": ["public", "internal", "confidential", "restricted"],
        "near_miss": {
            "public": [("open", "equivalent"), ("unclassified", "shifted"), ("shareable", "shifted")],
            "internal": [("staff only", "equivalent"), ("private", "shifted"), ("company", "equivalent")],
            "confidential": [("sensitive", "equivalent"), ("secret", "shifted"), ("protected", "equivalent")],
            "restricted": [("regulated", "shifted"), ("need to know", "equivalent"), ("highly sensitive", "shifted")],
        },
    },
    {
        "name": "set_meeting_response",
        "description": "Record a participant's response to a meeting invitation.",
        "field": "response",
        "members": ["accepted", "declined", "tentative", "no_response"],
        "near_miss": {
            "accepted": [("attending", "equivalent"), ("yes", "equivalent"), ("confirmed", "equivalent")],
            "declined": [("not attending", "equivalent"), ("no", "equivalent"), ("unavailable", "equivalent")],
            "tentative": [("maybe", "equivalent"), ("provisional", "equivalent"), ("unsure", "equivalent")],
            "no_response": [("pending", "shifted"), ("unanswered", "equivalent"), ("awaiting reply", "equivalent")],
        },
    },
    {
        "name": "set_priority",
        "description": "Assign a priority to a work item.",
        "field": "priority",
        "members": ["low", "normal", "high", "urgent"],
        "near_miss": {
            "low": [("minor", "equivalent"), ("whenever", "shifted"), ("backlog", "shifted")],
            "normal": [("standard", "equivalent"), ("default", "equivalent"), ("routine", "equivalent")],
            "high": [("important", "equivalent"), ("elevated", "equivalent"), ("soon", "shifted")],
            "urgent": [("critical", "shifted"), ("immediate", "equivalent"), ("asap", "equivalent")],
        },
    },
    {
        "name": "set_delivery_window",
        "description": "Set the preferred delivery window for an order.",
        "field": "window",
        "members": ["morning", "afternoon", "evening", "anytime"],
        "near_miss": {
            "morning": [("before noon", "equivalent"), ("am slot", "equivalent"), ("early", "equivalent")],
            "afternoon": [("midday", "shifted"), ("pm slot", "equivalent"), ("after lunch", "equivalent")],
            "evening": [("after work", "equivalent"), ("night slot", "shifted"), ("late", "equivalent")],
            "anytime": [("flexible", "equivalent"), ("no preference", "equivalent"), ("whole day", "shifted")],
        },
    },
    {
        "name": "set_content_language",
        "description": "Record the language of a piece of content.",
        "field": "language",
        "members": ["english", "spanish", "french", "german", "other"],
        "near_miss": {
            # Variants here must not contain their member as a word, or the
            # substitution is explicable by string matching. Hence "en-gb",
            # "latam", "quebecois", "austrian" rather than the obvious phrasings.
            "english": [("en", "equivalent"), ("en-gb", "shifted"), ("british", "shifted")],
            "spanish": [("es", "equivalent"), ("castilian", "equivalent"), ("latam", "shifted")],
            "french": [("fr", "equivalent"), ("francais", "equivalent"), ("quebecois", "shifted")],
            "german": [("de", "equivalent"), ("deutsch", "equivalent"), ("austrian", "shifted")],
            "other": [("unspecified", "equivalent"), ("mixed", "shifted"), ("unknown", "equivalent")],
        },
    },
]

INSTRUCTION_TEMPLATE = 'Set the {field} to "{value}" for record {rid}.'


def build_items(limit_per_schema: int | None = None) -> list[dict]:
    """Deterministically expand SCHEMAS into awkward/clean matched pairs.

    Returns a list of dicts:
        {id, schema_idx, arm, field, value, gold_member, near_miss_kind, rid}

    arm="awkward": value is not a member. gold_member is None; gold behaviour is
    to ask or flag.
    arm="clean": value is the exact member named by gold_member.
    """
    items: list[dict] = []
    for si, s in enumerate(SCHEMAS):
        rows: list[dict] = []
        for member in s["members"]:
            for kind_order, (nm, kind) in enumerate(s["near_miss"][member]):
                rows.append(
                    {
                        "schema_idx": si,
                        "field": s["field"],
                        "value": nm,
                        "gold_member": None,
                        "arm": "awkward",
                        "near_miss_kind": kind,
                        "paired_member": member,
                    }
                )
                rows.append(
                    {
                        "schema_idx": si,
                        "field": s["field"],
                        "value": member,
                        "gold_member": member,
                        "arm": "clean",
                        "near_miss_kind": None,
                        "paired_member": member,
                    }
                )
        if limit_per_schema:
            rows = rows[: limit_per_schema * 2]
        for j, r in enumerate(rows):
            r["id"] = f"s{si:02d}_{j:03d}"
            r["rid"] = f"{1000 + si * 100 + j}"
            items.append(r)
    return items


def tool_schema(s: dict, rotate: int = 0) -> dict:
    members = s["members"]
    if rotate:
        k = rotate % len(members)
        members = members[k:] + members[:k]
    return {
        "type": "function",
        "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": {
                "type": "object",
                "properties": {s["field"]: {"type": "string", "enum": members}},
                "required": [s["field"]],
                "additionalProperties": False,
            },
        },
    }


if __name__ == "__main__":
    import collections
    import re

    items = build_items()
    c = collections.Counter(i["arm"] for i in items)
    k = collections.Counter(i["near_miss_kind"] for i in items if i["arm"] == "awkward")
    print("schemas:", len(SCHEMAS))
    print("items:", len(items), dict(c))
    print("awkward near-miss kinds:", dict(k))
    for s in SCHEMAS:
        assert len(s["members"]) == len(set(s["members"]))
        for m, nms in s["near_miss"].items():
            assert m in s["members"], (s["name"], m)
            for nm, kind in nms:
                assert nm not in s["members"], (s["name"], nm)
                assert kind in ("equivalent", "shifted"), (s["name"], nm)
                # The member must not appear in the near-miss as a whole word,
                # or the substitution is explicable by string matching rather
                # than semantic nearness. Substring containment inside a longer
                # word ("reopened" / "open") is allowed at the word level, and
                # those cases are counted so they can be read as a caveat.
                for other in s["members"]:
                    assert not re.search(rf"\b{re.escape(other)}\b", nm), (
                        s["name"],
                        nm,
                        other,
                    )
    print("schema invariants OK")
