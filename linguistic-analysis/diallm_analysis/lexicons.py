"""Curated, high-precision dialectal lexicons and British/American spelling pairs.

These are *independent* of the eWAVE feature classifier used for the training
reward: they are hand-curated surface markers, so counting them does not re-use
the reward signal. Lexicons favour precision over recall (they are a lower bound
on dialectal marking, not an exhaustive inventory).

variety keys: 'aus' (en-AU), 'ind' (en-IN), 'uk' (en-UK / Northern British).
"""

# ---------------------------------------------------------------------------
# Lexical markers (single tokens and multiword expressions).
# Kept variety-discriminative; shared general-colloquial items are avoided
# where they would blur attribution.
# ---------------------------------------------------------------------------
LEXICON = {
    "aus": {
        "tokens": {
            "arvo", "servo", "brekkie", "brekky", "barbie", "bottlo", "mozzie",
            "mozzies", "sunnies", "ute", "esky", "tradie", "tradies", "maccas",
            "footy", "doona", "dunny", "chook", "chooks", "bikkie", "bikkies",
            "snag", "snags", "sanger", "stoked", "ripper", "bogan", "daggy",
            "sook", "chunder", "dunny", "arvos", "servos", "smoko", "tinnie",
            "tinnies", "thongs", "togs", "rego", "yobbo", "dunno",
        },
        "phrases": {
            "fair dinkum", "no worries", "g'day", "good on ya", "good on you",
            "how ya going", "heaps good", "too easy", "she'll be right",
        },
    },
    "ind": {
        "tokens": {
            "prepone", "preponed", "updation", "upgradation", "intimation",
            "batchmate", "batchmates", "timepass", "lakh", "lakhs", "crore",
            "crores", "tiffin", "brinjal", "capsicum", "ladyfinger", "curd",
            "pucca", "kaccha", "kacha", "jugaad", "yaar", "achcha", "na",
            "needful", "revert", "marriage", "function", "prepone",
        },
        "phrases": {
            "do the needful", "good name", "out of station", "pass out",
            "passing out", "cousin brother", "cousin sister", "real brother",
            "real sister", "co-brother", "time pass", "kindly do",
            "please do the needful", "mention not", "what is your good name",
            "do one thing", "discuss about", "comprises of", "comprised of",
            "stress on", "cope up", "revert back", "return back", "order for",
        },
    },
    "uk": {  # Northern British leaning
        "tokens": {
            "nowt", "owt", "summat", "aye", "lass", "lasses", "lad", "lads",
            "bairn", "bairns", "mam", "mardy", "ginnel", "snicket", "mither",
            "mithering", "chuffed", "gutted", "lush", "reet", "ta", "brew",
            "butty", "barm", "barmcake", "nesh", "scran", "grafting", "graft",
            "canny", "wor", "howay", "gan", "summ'at", "tret", "mam",
        },
        "phrases": {
            "ey up", "our lass", "our kid", "proper good", "made up",
            "i'm made up", "going the match", "round ours",
        },
    },
}

# ---------------------------------------------------------------------------
# British vs American spelling pairs (orthographic markers).
# High precision: explicit word pairs rather than suffix heuristics, so the
# pronoun "our" / "hour" etc. cannot be mis-counted as a "-our" British form.
# Counting British spellings is a classifier-independent en-UK / en-AU signal,
# and is the cleanest objective marker for en-UK (where the variety classifier
# collapsed in the paper).
# ---------------------------------------------------------------------------
BRITISH_SPELLINGS = {
    # -our / -or
    "colour", "colours", "coloured", "colouring", "flavour", "flavours",
    "flavoured", "behaviour", "behaviours", "neighbour", "neighbours",
    "favour", "favours", "favourite", "favourites", "honour", "honours",
    "labour", "humour", "humours", "rumour", "rumours", "harbour", "vapour",
    "odour", "savour", "savoury", "endeavour", "splendour", "valour",
    # -re / -er
    "centre", "centres", "theatre", "theatres", "metre", "metres", "litre",
    "litres", "fibre", "fibres", "calibre", "sombre", "spectre", "lustre",
    "manoeuvre", "meagre",
    # -ise / -isation (and -yse)
    "realise", "realised", "realising", "organise", "organised", "organising",
    "organisation", "recognise", "recognised", "apologise", "apologised",
    "analyse", "analysed", "criticise", "criticised", "emphasise",
    "emphasised", "prioritise", "prioritised", "specialise", "minimise",
    "maximise", "summarise", "categorise", "memorise", "customise",
    "optimise", "civilisation", "globalisation",
    # doubled l
    "travelling", "travelled", "traveller", "cancelled", "cancelling",
    "labelled", "labelling", "modelling", "modelled", "fuelled", "marvellous",
    "jewellery", "counsellor",
    # -ogue / misc
    "catalogue", "dialogue", "analogue", "grey", "programme", "whilst",
    "amongst", "learnt", "dreamt", "spelt", "towards", "maths", "aluminium",
    "defence", "offence", "licence", "practise", "cheque", "kerb", "tyre",
    "tyres", "pyjamas", "mum", "mummy", "aeroplane", "storey", "plough",
    "draught", "sceptical", "moustache", "cosy", "doughnut",
}

AMERICAN_SPELLINGS = {
    "color", "colors", "colored", "coloring", "flavor", "flavors", "flavored",
    "behavior", "behaviors", "neighbor", "neighbors", "favor", "favors",
    "favorite", "favorites", "honor", "honors", "labor", "humor", "rumor",
    "rumors", "harbor", "vapor", "odor", "savor", "savory", "endeavor",
    "splendor", "valor",
    "center", "centers", "theater", "theaters", "meter", "meters", "liter",
    "liters", "fiber", "fibers", "caliber", "somber", "specter", "luster",
    "maneuver", "meager",
    "realize", "realized", "realizing", "organize", "organized", "organizing",
    "organization", "recognize", "recognized", "apologize", "apologized",
    "analyze", "analyzed", "criticize", "criticized", "emphasize",
    "emphasized", "prioritize", "prioritized", "specialize", "minimize",
    "maximize", "summarize", "categorize", "memorize", "customize", "optimize",
    "civilization", "globalization",
    "traveling", "traveled", "traveler", "canceled", "canceling", "labeled",
    "labeling", "modeling", "modeled", "fueled", "marvelous", "jewelry",
    "counselor",
    "catalog", "dialog", "analog", "gray", "program", "math", "aluminum",
    "defense", "offense", "license", "practice", "check", "curb", "tire",
    "tires", "pajamas", "mom", "mommy", "airplane", "story", "plow", "draft",
    "skeptical", "mustache", "cozy", "donut",
}

# Stative verbs for the progressive-with-stative detector (en-IN feature).
STATIVE_VERBS = {
    "understand", "know", "believe", "want", "need", "like", "love", "hate",
    "prefer", "mean", "have", "own", "possess", "contain", "doubt", "remember",
    "realise", "realize", "see", "hear", "seem", "belong", "suppose", "wish",
    "recognise", "recognize", "deserve", "consist",
}

# Politeness / hedge markers (register; en-IN tends to over-mark politeness).
POLITENESS_MARKERS = {
    "kindly", "please", "perhaps", "maybe", "sorry", "thanks", "thank",
    "would", "could", "request", "humbly", "respected", "obliged", "grateful",
}
