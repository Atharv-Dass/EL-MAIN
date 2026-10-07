"""Surface templates ("families") that turn a computation DAG into a word problem.

Each family is one story world. Every operation template has exactly one
arithmetic meaning, so the text and the DAG cannot disagree.

Placeholders:
    {who}  - the actor whose quantity is being tracked
    {n}    - the operand number for this step
    {other}, {m} - distractor-only actor and number

`split="heldout"` families are NEVER used for training data. They use different
phrasing on purpose, so gains that come from memorising templates show up as
a train/held-out gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dag import Op

NAMES: tuple[str, ...] = (
    "Asha", "Ravi", "Maya", "Leo", "Priya", "Sam", "Noor", "Kiran", "Elena", "Tariq",
    "Mei", "Jonah", "Aditi", "Omar", "Zara", "Felix", "Ishaan", "Lucia", "Dev", "Hana",
)


@dataclass(frozen=True)
class Family:
    name: str
    split: str  # "train" or "heldout"
    unit: str   # noun used in questions, e.g. "apples" or "money"
    money: bool
    start: tuple[str, ...]
    ops: dict[Op, tuple[str, ...]]
    question: tuple[str, ...]
    question_sum: tuple[str, ...]   # merge with ADD: {who} and {who2}
    question_diff: tuple[str, ...]  # merge with SUB: {who} has more than {who2}
    distractors: tuple[str, ...] = field(default_factory=tuple)


def _count_family(name: str, unit: str, split: str, add: tuple[str, ...], sub: tuple[str, ...],
                  mul: tuple[str, ...], div: tuple[str, ...], start: tuple[str, ...],
                  distractors: tuple[str, ...],
                  question: tuple[str, ...] | None = None,
                  question_sum: tuple[str, ...] | None = None,
                  question_diff: tuple[str, ...] | None = None) -> Family:
    return Family(
        name=name,
        split=split,
        unit=unit,
        money=False,
        start=start,
        ops={Op.ADD: add, Op.SUB: sub, Op.MUL: mul, Op.DIV: div},
        question=question or (f"How many {unit} does {{who}} have now?",
                              f"How many {unit} does {{who}} end up with?"),
        question_sum=question_sum or (f"How many {unit} do {{who}} and {{who2}} have altogether?",),
        question_diff=question_diff or (f"How many more {unit} does {{who}} have than {{who2}}?",),
        distractors=distractors,
    )


FAMILIES: dict[str, Family] = {}


def _register(f: Family) -> None:
    FAMILIES[f.name] = f


# ----------------------------------------------------------------- train split
_register(_count_family(
    "fruit_stand", "apples", "train",
    start=("{who} has {n} apples at a fruit stand.", "{who} starts the day with {n} apples."),
    add=("{who} buys {n} more apples from a farmer.", "A supplier delivers {n} apples to {who}."),
    sub=("{who} sells {n} apples.", "{who} gives {n} apples to a neighbour."),
    mul=("After a big harvest, {who}'s number of apples becomes {n} times as large.",),
    div=("{who} splits the apples equally into {n} baskets and keeps only one basket.",),
    distractors=("{other} has {m} oranges.", "The fruit stand opened {m} days ago."),
))
_register(_count_family(
    "bakery", "cookies", "train",
    start=("{who} bakes {n} cookies.", "{who} has {n} cookies in the bakery."),
    add=("{who} bakes another {n} cookies.", "{who} receives {n} cookies from a friend."),
    sub=("{who} sells {n} cookies.", "{who} eats {n} of the cookies."),
    mul=("{who} scales up the batch so the number of cookies becomes {n} times as large.",),
    div=("{who} packs the cookies equally into {n} boxes and keeps just one box.",),
    distractors=("{other} bakes {m} muffins.", "The oven is set to {m} degrees."),
))
_register(_count_family(
    "library", "books", "train",
    start=("{who} has {n} books on a shelf.", "{who} owns {n} books."),
    add=("{who} buys {n} new books.", "{who} is given {n} books as a gift."),
    sub=("{who} donates {n} books.", "{who} lends out {n} books and does not get them back."),
    mul=("{who}'s book collection then grows to {n} times its size.",),
    div=("{who} divides the books equally among {n} shelves and keeps only one shelf of them.",),
    distractors=("{other} has read {m} magazines.", "The library is {m} years old."),
))
_register(_count_family(
    "farm", "eggs", "train",
    start=("{who} collects {n} eggs.", "{who} has {n} eggs in the barn."),
    add=("{who} collects {n} more eggs.", "The hens lay another {n} eggs for {who}."),
    sub=("{who} uses {n} eggs for breakfast.", "{who} sells {n} eggs at the market."),
    mul=("By the end of the season, {who}'s number of eggs becomes {n} times as large.",),
    div=("{who} places the eggs equally into {n} cartons and keeps one carton.",),
    distractors=("{other} owns {m} cows.", "The farm is {m} acres."),
))
_register(Family(
    name="piggy_bank", split="train", unit="money", money=True,
    start=("{who} has ${n} in a piggy bank.", "{who} saves ${n}."),
    ops={
        Op.ADD: ("{who} earns ${n} doing chores.", "{who} receives ${n} as a gift."),
        Op.SUB: ("{who} spends ${n} on a toy.", "{who} pays ${n} for lunch."),
        Op.MUL: ("{who}'s savings then grow to {n} times the amount.",),
        Op.DIV: ("{who} splits the money equally into {n} envelopes and keeps one envelope.",),
    },
    question=("How much money does {who} have now?", "How many dollars does {who} end up with?"),
    question_sum=("How much money do {who} and {who2} have altogether?",),
    question_diff=("How much more money does {who} have than {who2}?",),
    distractors=("{other} has ${m} in a wallet.", "The piggy bank weighs {m} grams."),
))
_register(_count_family(
    "sticker_album", "stickers", "train",
    start=("{who} has {n} stickers.", "{who}'s album holds {n} stickers."),
    add=("{who} trades for {n} more stickers.", "{who} buys a pack of {n} stickers."),
    sub=("{who} gives away {n} stickers.", "{who} loses {n} stickers."),
    mul=("{who}'s sticker count then becomes {n} times as large.",),
    div=("{who} shares the stickers equally among {n} friends, keeping one friend's share.",),
    distractors=("{other} has {m} postcards.", "The album has {m} pages."),
))
_register(_count_family(
    "marble_jar", "marbles", "train",
    start=("{who} has {n} marbles in a jar.", "{who} owns {n} marbles."),
    add=("{who} wins {n} marbles in a game.", "{who} finds {n} more marbles."),
    sub=("{who} loses {n} marbles in a game.", "{who} gives {n} marbles to a cousin."),
    mul=("{who}'s marble count then becomes {n} times as large.",),
    div=("{who} sorts the marbles equally into {n} bags and keeps one bag.",),
    distractors=("{other} has {m} toy cars.", "The jar is {m} centimetres tall."),
))
_register(_count_family(
    "seed_store", "seed packets", "train",
    start=("{who} has {n} seed packets.", "{who} stocks {n} seed packets."),
    add=("{who} orders {n} more seed packets.", "{who} receives a shipment of {n} seed packets."),
    sub=("{who} sells {n} seed packets.", "{who} plants {n} seed packets."),
    mul=("{who}'s stock of seed packets then becomes {n} times as large.",),
    div=("{who} divides the seed packets equally among {n} gardens and keeps one garden's share.",),
    distractors=("{other} has {m} flower pots.", "The store is open {m} hours a week."),
))

# -------------------------------------------------------------- held-out split
_register(_count_family(
    "warehouse", "crates", "heldout",
    start=("The warehouse managed by {who} holds {n} crates.",
           "At the start of the month, {who}'s warehouse contains {n} crates."),
    add=("A truck unloads a further {n} crates into {who}'s warehouse.",
         "{who} takes delivery of {n} additional crates."),
    sub=("{n} crates are shipped out of {who}'s warehouse.",
         "{who} dispatches {n} crates to customers."),
    mul=("Following an expansion, the number of crates at {who}'s warehouse is multiplied by {n}.",),
    div=("{who} distributes the crates evenly across {n} loading bays and only the first bay's "
         "crates remain.",),
    distractors=("A forklift at {other}'s warehouse can lift {m} kilograms.",
                 "The warehouse has {m} employees."),
    question=("How many crates are in {who}'s warehouse at the end?",),
    question_sum=("What is the combined number of crates in the warehouses of {who} and {who2}?",),
    question_diff=("By how many crates does {who}'s warehouse exceed {who2}'s?",),
))
_register(Family(
    name="fundraiser", split="heldout", unit="money", money=True,
    start=("A fundraiser organised by {who} begins with ${n}.",
           "{who}'s fundraiser has raised ${n} so far."),
    ops={
        Op.ADD: ("Donors contribute an additional ${n} to {who}'s fundraiser.",
                 "{who}'s fundraiser collects another ${n} at an event."),
        Op.SUB: ("{who}'s fundraiser pays ${n} in expenses.",
                 "The fundraiser run by {who} spends ${n} on posters."),
        Op.MUL: ("A sponsor multiplies the total in {who}'s fundraiser by {n}.",),
        Op.DIV: ("The total in {who}'s fundraiser is divided equally among {n} charities, and "
                 "{who} tracks only one charity's portion.",),
    },
    question=("How many dollars does {who}'s fundraiser have at the end?",),
    question_sum=("How many dollars do the fundraisers of {who} and {who2} hold combined?",),
    question_diff=("By how many dollars does {who}'s fundraiser exceed {who2}'s?",),
    distractors=("{other} volunteered for {m} hours.", "The event hall seats {m} people."),
))


def families_for_split(split: str) -> list[Family]:
    fams = [f for f in FAMILIES.values() if f.split == split]
    if not fams:
        raise ValueError(f"no families for split {split!r}")
    return fams
