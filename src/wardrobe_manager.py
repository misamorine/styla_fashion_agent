import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from PIL import Image

from .catalog_utils import infer_slot, infer_color, infer_gender, infer_occasion_group, safe_text
from .rules import score_item


class WardrobeManager:
    """Simple local wardrobe database for the MVP.

    Stores user-owned items in JSON and saves uploaded images locally.
    This is intentionally lightweight so it works without a server/database.
    """

    def __init__(self, db_path: str = "./wardrobe_db.json", image_dir: str = "./wardrobe_images"):
        self.db_path = Path(db_path)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write([])

    def _read(self) -> List[Dict]:
        try:
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, rows: List[Dict]) -> None:
        self.db_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_items(self) -> List[Dict]:
        return self._read()

    def count(self) -> int:
        return len(self._read())

    def add_item(
        self,
        image_file,
        title: str,
        description: str = "",
        gender: str = "auto",
        slot: str = "auto",
        color: str = "auto",
        occasion_group: str = "auto",
    ) -> Dict:
        item_id = f"wardrobe_{uuid4().hex[:10]}"
        image_path = ""
        if image_file is not None:
            image_path = str(self.image_dir / f"{item_id}.jpg")
            img = Image.open(image_file).convert("RGB")
            img.save(image_path, quality=90)

        combined = f"{title} {description}"
        inferred_slot = infer_slot(combined)
        inferred_color = infer_color(combined)
        inferred_gender = infer_gender(combined)
        inferred_occasion = infer_occasion_group(combined)

        row = {
            "id": item_id,
            "source": "wardrobe",
            "owned": True,
            "title": safe_text(title) or "Wardrobe item",
            "description": safe_text(description) or safe_text(combined),
            "category": slot if slot != "auto" else inferred_slot,
            "slot": slot if slot != "auto" else inferred_slot,
            "color": color if color != "auto" else inferred_color,
            "gender": gender if gender != "auto" else inferred_gender,
            "occasion_group": occasion_group if occasion_group != "auto" else inferred_occasion,
            "image_path": image_path,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        rows = self._read()
        rows.append(row)
        self._write(rows)
        return row

    def delete_item(self, item_id: str) -> bool:
        rows = self._read()
        kept = [r for r in rows if r.get("id") != item_id]
        deleted = len(kept) != len(rows)
        if deleted:
            item = next((r for r in rows if r.get("id") == item_id), None)
            img = item.get("image_path") if item else None
            if img:
                try:
                    Path(img).unlink(missing_ok=True)
                except Exception:
                    pass
            self._write(kept)
        return deleted

    def clear(self) -> None:
        self._write([])
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def candidates_for_slot(self, intent, slot: str) -> List[Dict]:
        """Return wardrobe candidates matching a slot, ranked by existing rule scorer."""
        candidates = []
        for row in self._read():
            if row.get("slot") != slot:
                continue
            gender = row.get("gender")
            if intent.gender in ["male", "female"] and gender not in [intent.gender, "unisex", "unknown"]:
                continue
            candidate = dict(row)
            candidate["used_where_filter"] = "local wardrobe JSON; no ChromaDB filter"
            candidate["rule_score"] = score_item(candidate, intent, slot)
            candidate["vector_score"] = 0.0
            # Owned bonus makes the stylist prefer clothes user already owns.
            candidate["final_score"] = candidate["rule_score"] + 18.0
            candidates.append(candidate)
        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return candidates
