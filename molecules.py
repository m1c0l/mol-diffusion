"""
50 common drug-like molecules as SMILES strings.
Covers a range of sizes (5-30 heavy atoms), ring systems, heteroatoms.
"""

SMILES_LIST = [
    # Small/simple
    ("water",           "O"),
    ("ethanol",         "CCO"),
    ("acetic_acid",     "CC(=O)O"),
    ("acetone",         "CC(=O)C"),
    ("benzene",         "c1ccccc1"),
    ("toluene",         "Cc1ccccc1"),
    ("phenol",          "Oc1ccccc1"),
    ("aniline",         "Nc1ccccc1"),
    ("pyridine",        "c1ccncc1"),
    ("imidazole",       "c1cnc[nH]1"),

    # Amino acids
    ("glycine",         "NCC(=O)O"),
    ("alanine",         "CC(N)C(=O)O"),
    ("phenylalanine",   "NC(Cc1ccccc1)C(=O)O"),
    ("tryptophan",      "NC(Cc1c[nH]c2ccccc12)C(=O)O"),
    ("histidine",       "NC(Cc1cnc[nH]1)C(=O)O"),

    # Common drugs / drug fragments
    ("aspirin",         "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol",     "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen",       "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine",        "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("dopamine",        "NCCc1ccc(O)c(O)c1"),
    ("serotonin",       "NCCc1c[nH]c2ccc(O)cc12"),
    ("nicotine",        "CN1CCC[C@H]1c1cccnc1"),
    ("lidocaine",       "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("metformin",       "CN(C)C(=N)NC(=N)N"),
    ("penicillin_g",    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),

    # Nucleobases
    ("adenine",         "Nc1ncnc2[nH]cnc12"),
    ("guanine",         "Nc1nc2[nH]cnc2c(=O)[nH]1"),
    ("cytosine",        "Nc1ccnc(=O)[nH]1"),
    ("thymine",         "Cc1cnc(=O)[nH]c1=O"),
    ("uracil",          "O=c1ccnc(=O)[nH]1"),

    # Lipid/fatty acid fragments
    ("choline",         "C[N+](C)(C)CCO"),
    ("glycerol",        "OCC(O)CO"),

    # More ring systems
    ("indole",          "c1ccc2[nH]ccc2c1"),
    ("quinoline",       "c1ccc2ncccc2c1"),
    ("furan",           "c1ccoc1"),
    ("thiophene",       "c1ccsc1"),
    ("morpholine",      "C1COCCN1"),
    ("piperidine",      "C1CCNCC1"),
    ("piperazine",      "C1CNCCN1"),

    # Vitamins / cofactors (fragments)
    ("niacin",          "OC(=O)c1cccnc1"),
    ("pantothenic_frag","CC(C)(CO)C(O)C(=O)NCC(=O)O"),
    ("biotin_frag",     "O=C1NC(=O)[C@@H]2CS[C@@H](CCCCC(=O)O)[C@@H]2N1"),

    # Common pharmacophores
    ("sulfonamide_ex",  "Cc1ccc(cc1)S(=O)(=O)N"),
    ("urea_ex",         "NC(=O)N"),
    ("guanidine",       "NC(=N)N"),
    ("hydroxamic_acid", "CC(=O)NO"),
    ("beta_lactam",     "O=C1CCN1"),

    # Steroids / terpenoids (small representatives)
    ("menthol",         "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"),
    ("limonene",        "CC(=C)[C@@H]1CCC(=C)CC1"),

    # Misc
    ("epinephrine",     "CNC[C@@H](O)c1ccc(O)c(O)c1"),
    ("histamine",       "NCCc1c[nH]cn1"),
    ("GABA",            "NCCCC(=O)O"),
]
