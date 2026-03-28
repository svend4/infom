"""
16 информационных архетипов.
Портировано из pseudorag/core/archetypes.py (svend4/daten22)

4 оси измерения:
    M/A — Материальность (Material / Abstract)
    S/D — Динамика (Static / Dynamic)
    E/C — Масштаб (Elementary / Complex)
    O/F — Структура (Ordered / Fluid)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Archetype:
    code:        str           # "MSEO", "MDEF", ...
    name_ru:     str
    name_en:     str
    description_ru: str
    description_en: str
    keywords_ru: list[str]
    keywords_en: list[str]
    examples:    list[str]
    priority:    int           # 1–5

    @property
    def quadrant(self) -> str:
        return self.code[:2]   # "MS", "MD", "AS", "AD"

    @property
    def is_material(self) -> bool:  return self.code[0] == "M"
    @property
    def is_dynamic(self)  -> bool:  return self.code[1] == "D"
    @property
    def is_complex(self)  -> bool:  return self.code[2] == "C"
    @property
    def is_ordered(self)  -> bool:  return self.code[3] == "O"

    def to_hex_bits(self) -> tuple[int,...]:
        """Маппинг архетипа в 4-битный вектор (часть Q6)."""
        return (
            1 if self.is_material else 0,
            1 if self.is_dynamic  else 0,
            1 if self.is_complex  else 0,
            1 if self.is_ordered  else 0,
        )


ARCHETYPES: list[Archetype] = [
    # ── Quadrant I: Material-Static (MS) ─────────────────────────────────
    Archetype(
        code="MSEO", name_ru="Кристалл", name_en="Crystal",
        description_ru="Простые упорядоченные физические структуры",
        description_en="Simple ordered physical structures",
        keywords_ru=["структура", "порядок", "решётка", "форма", "симметрия"],
        keywords_en=["structure", "order", "lattice", "form", "symmetry"],
        examples=["кристалл соли", "снежинка", "минерал", "атом"],
        priority=3,
    ),
    Archetype(
        code="MSEF", name_ru="Песок", name_en="Sand",
        description_ru="Простые неупорядоченные физические частицы",
        description_en="Simple disordered physical particles",
        keywords_ru=["частицы", "рассеяние", "хаос", "зерно", "порошок"],
        keywords_en=["particles", "scatter", "chaos", "grain", "powder"],
        examples=["песок", "пыль", "гравий", "капли воды"],
        priority=2,
    ),
    Archetype(
        code="MSCO", name_ru="Здание", name_en="Building",
        description_ru="Сложные упорядоченные материальные системы",
        description_en="Complex ordered material systems",
        keywords_ru=["здание", "конструкция", "архитектура", "инфраструктура"],
        keywords_en=["building", "construction", "architecture", "infrastructure"],
        examples=["небоскрёб", "мост", "завод", "плотина"],
        priority=4,
    ),
    Archetype(
        code="MSCF", name_ru="Лес", name_en="Forest",
        description_ru="Сложные природные экосистемы",
        description_en="Complex natural ecosystems",
        keywords_ru=["экосистема", "природа", "биом", "ландшафт"],
        keywords_en=["ecosystem", "nature", "biome", "landscape"],
        examples=["тропический лес", "коралловый риф", "тундра", "болото"],
        priority=3,
    ),

    # ── Quadrant II: Material-Dynamic (MD) ───────────────────────────────
    Archetype(
        code="MDEO", name_ru="Механизм", name_en="Mechanism",
        description_ru="Простые упорядоченные движущиеся устройства",
        description_en="Simple ordered moving devices",
        keywords_ru=["механизм", "шестерня", "рычаг", "привод", "движение"],
        keywords_en=["mechanism", "gear", "lever", "drive", "motion"],
        examples=["часы", "замок", "пружина", "маятник"],
        priority=3,
    ),
    Archetype(
        code="MDEF", name_ru="Организм", name_en="Organism",
        description_ru="Простые адаптивные живые существа",
        description_en="Simple adaptive living creatures",
        keywords_ru=["организм", "клетка", "бактерия", "адаптация", "жизнь"],
        keywords_en=["organism", "cell", "bacteria", "adaptation", "life"],
        examples=["бактерия", "вирус", "амёба", "водоросль"],
        priority=4,
    ),
    Archetype(
        code="MDCO", name_ru="Машина", name_en="Machine",
        description_ru="Сложные технические системы",
        description_en="Complex technical systems",
        keywords_ru=["машина", "двигатель", "система", "технология", "автомат"],
        keywords_en=["machine", "engine", "system", "technology", "automaton"],
        examples=["автомобиль", "самолёт", "компьютер", "робот"],
        priority=5,
    ),
    Archetype(
        code="MDCF", name_ru="Город", name_en="City",
        description_ru="Сложные органические динамические системы",
        description_en="Complex organic dynamic systems",
        keywords_ru=["город", "транспорт", "население", "урбанизация", "сеть"],
        keywords_en=["city", "transport", "population", "urbanization", "network"],
        examples=["мегаполис", "рынок", "экосистема города", "социум"],
        priority=5,
    ),

    # ── Quadrant III: Abstract-Static (AS) ───────────────────────────────
    Archetype(
        code="ASEO", name_ru="Аксиома", name_en="Axiom",
        description_ru="Фундаментальные неизменные истины",
        description_en="Fundamental unchanging truths",
        keywords_ru=["аксиома", "принцип", "закон", "теорема", "постулат"],
        keywords_en=["axiom", "principle", "law", "theorem", "postulate"],
        examples=["закон Ньютона", "аксиомы геометрии", "логический закон"],
        priority=4,
    ),
    Archetype(
        code="ASEF", name_ru="Архетип", name_en="Archetype",
        description_ru="Базовые неструктурированные паттерны",
        description_en="Basic unstructured patterns",
        keywords_ru=["архетип", "образ", "символ", "прообраз", "паттерн"],
        keywords_en=["archetype", "image", "symbol", "prototype", "pattern"],
        examples=["герой", "тень", "анима", "трикстер"],
        priority=3,
    ),
    Archetype(
        code="ASCO", name_ru="Теория", name_en="Theory",
        description_ru="Структурированные системы знания",
        description_en="Structured knowledge systems",
        keywords_ru=["теория", "модель", "концепция", "фреймворк", "наука"],
        keywords_en=["theory", "model", "concept", "framework", "science"],
        examples=["квантовая механика", "теория эволюции", "теория графов"],
        priority=5,
    ),
    Archetype(
        code="ASCF", name_ru="Культура", name_en="Culture",
        description_ru="Сложные системы ценностей и традиций",
        description_en="Complex value and tradition systems",
        keywords_ru=["культура", "традиция", "ценности", "норма", "менталитет"],
        keywords_en=["culture", "tradition", "values", "norm", "mentality"],
        examples=["японская культура", "религия", "субкультура", "язык"],
        priority=4,
    ),

    # ── Quadrant IV: Abstract-Dynamic (AD) ───────────────────────────────
    Archetype(
        code="ADEO", name_ru="Алгоритм", name_en="Algorithm",
        description_ru="Детерминированные пошаговые процедуры",
        description_en="Deterministic step-by-step procedures",
        keywords_ru=["алгоритм", "процедура", "шаг", "инструкция", "протокол"],
        keywords_en=["algorithm", "procedure", "step", "instruction", "protocol"],
        examples=["сортировка", "шифрование", "рецепт", "маршрут"],
        priority=5,
    ),
    Archetype(
        code="ADEF", name_ru="Интуиция", name_en="Intuition",
        description_ru="Спонтанные мыслительные процессы",
        description_en="Spontaneous thought processes",
        keywords_ru=["интуиция", "чувство", "инсайт", "эмоция", "спонтанность"],
        keywords_en=["intuition", "feeling", "insight", "emotion", "spontaneity"],
        examples=["творческое озарение", "эмпатия", "предчувствие"],
        priority=2,
    ),
    Archetype(
        code="ADCO", name_ru="Программа", name_en="Program",
        description_ru="Сложные алгоритмические системы",
        description_en="Complex algorithmic systems",
        keywords_ru=["программа", "система", "код", "архитектура", "платформа"],
        keywords_en=["program", "system", "code", "architecture", "platform"],
        examples=["операционная система", "нейросеть", "ERP", "браузер"],
        priority=5,
    ),
    Archetype(
        code="ADCF", name_ru="Общество", name_en="Society",
        description_ru="Сложная социальная динамика",
        description_en="Complex social dynamics",
        keywords_ru=["общество", "социум", "сообщество", "движение", "политика"],
        keywords_en=["society", "community", "movement", "politics", "social"],
        examples=["государство", "рынок", "интернет-сообщество", "революция"],
        priority=4,
    ),
]

# ── утилиты ─────────────────────────────────────────────────────────────────

ARCHETYPE_MAP: dict[str, Archetype] = {a.code: a for a in ARCHETYPES}


def get_archetype(code: str) -> Archetype | None:
    return ARCHETYPE_MAP.get(code)


def get_by_quadrant(quadrant: str) -> list[Archetype]:
    """quadrant: "MS", "MD", "AS", "AD" """
    return [a for a in ARCHETYPES if a.quadrant == quadrant]


def get_by_priority(min_priority: int = 4) -> list[Archetype]:
    return [a for a in ARCHETYPES if a.priority >= min_priority]


def find_by_keyword(keyword: str) -> list[Archetype]:
    kw = keyword.lower()
    return [
        a for a in ARCHETYPES
        if any(kw in k.lower() for k in a.keywords_ru + a.keywords_en)
    ]


def archetype_to_hex_id(code: str) -> int:
    """Маппинг архетипа в Q6 id (первые 4 бита)."""
    a = get_archetype(code)
    if not a:
        return 0
    bits = a.to_hex_bits()
    return sum(b << i for i, b in enumerate(bits))
