
# Central registry of AGP metadata features with completeness > 60%.
#
# Feature selection aims to establish a questionnaire-based baseline
# without using diagnosed conditions or features likely to encode
# cohort/recruitment effects.


# --------------------------------------------------------------------------
# Target / grouping (never use as features)
# --------------------------------------------------------------------------
TARGET = "ibd"
GROUP_COL = "host_subject_id"


# --------------------------------------------------------------------------
# Numeric features
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "host_age",
    "host_height",
    "host_weight",
]


# --------------------------------------------------------------------------
# Fermented food items; the same 21 items exist for three question blocks
# (consumed / produce_commercial / produce_personal).
# Comment items out here to remove them from every block at once.
# --------------------------------------------------------------------------
_FERMENTED_ITEMS = [
    "beer",
    "chicha",
    "cider",
    "cottage_cheese",
    "fermented_beansmisonatto",
    "fermented_breadsourdoughinjera",
    "fermented_fish",
    "fermented_tofu",
    "fish_sauce",
    "kefir_milk",
    "kefir_water",
    "kimchi",
    "kombucha",
    "mead",
    "pickled_vegetables",
    "sauerkraut",
    "sour_creamcreme_fraiche",
    "tempeh",
    # "unspecified",   # free-text "other" flag, almost always empty
    "wine",
    "yogurtlassi",
]


# --------------------------------------------------------------------------
# Categorical feature registry: CATEGORY -> list of features
# Numeric features are defined separately above.
# --------------------------------------------------------------------------
CATEGORIES: dict[str, list[str]] = {

    # ----------------------------------------------------------------------
    "DEMOGRAPHICS": [
        "sex",
        # "race",                     # excluded: potential cohort/recruitment effects
        # "country_of_birth",         # excluded: potential cohort/recruitment effects
        # "weight_change",            # unclear interpretation / temporal reference
    ],

    # ----------------------------------------------------------------------
    "SAMPLE": [
        # "host_body_mass_index",     # derived from height/weight; avoid redundant representation
        # "bmi_cat",                  # derived from height/weight; loses information through binning
        # "body_site",                # redundant: same info as host_body_site (filtered to feces)
        # "host_body_site",           # constant after filtering to uberon:feces
        # "host_body_habitat",        # constant
        # "host_body_product",        # constant
        # "host_common_name",         # constant
        # "host_taxid",               # constant
        # "scientific_name",          # constant
        # "taxon_id",                 # constant
        # "dna_extracted",             # lab metadata, not a host characteristic
        # "physical_specimen_remaining",  # lab metadata, not a host characteristic
    ],

    # ----------------------------------------------------------------------
    "LIFESTYLE": [
        # "alcohol_consumption",      # less detailed than alcohol_frequency
        "alcohol_frequency",
        "alcohol_types_beercider",
        "alcohol_types_red_wine",
        "alcohol_types_sour_beers",
        "alcohol_types_spiritshard_alcohol",
        # "alcohol_types_unspecified", # uninformative "other" flag
        "alcohol_types_white_wine",
        "exercise_frequency",
        # "exercise_location",        # limited additional information
        # "pool_frequency",            # limited additional information
        "sleep_duration",
        "smoking_frequency",
    ],

    # ----------------------------------------------------------------------
    "LIFESTYLE_MISC": [
        # "dominant_hand",             # no clear relevance to the prediction task
        # "last_move",                 # limited relevance; potentially cohort-specific
        "last_travel",
        "livingwith",
        # "nail_biter",                # no clear relevance to the prediction task
        # "roommates_in_study",        # study-specific, 26 % missing
        # "softener",                  # no clear relevance to the prediction task
        "types_of_plants",
        # "vivid_dreams",              # no clear relevance; 32 % missing
    ],

    # ----------------------------------------------------------------------
    "HYGIENE": [
        # "cosmetics_frequency",       # no clear relevance to the prediction task
        # "deodorant_use",             # no clear relevance to the prediction task
        "flossing_frequency",
        "teethbrushing_frequency",
    ],

    # ----------------------------------------------------------------------
    "PETS_AND_SOCIAL": [
        "cat",
        "dog",
        "pets_other",
        # "level_of_education",        # possible socioeconomic proxy; not included in current baseline
    ],

    # ----------------------------------------------------------------------
    "ENVIRONMENT": [
        "drinking_water_source",
    ],

    # ----------------------------------------------------------------------
    "LOCATION": [
        # "state",                     # excluded: potential cohort/recruitment effects, high cardinality
        # "country_residence",         # excluded: potential cohort/recruitment effects, 30 % missing
        # "geo_loc_name",              # excluded: high cardinality, geographic/cohort effects
        # "latitude",                  # excluded: geographic proxy and quasi-identifier
        # "longitude",                 # excluded: geographic proxy and quasi-identifier
        # "env_biome",                 # constant
        # "env_feature",               # constant
        # "env_material",              # constant
        # "env_package",               # constant
    ],

    # ----------------------------------------------------------------------
    # DISEASE: other diagnosed conditions.
    # Excluded to avoid using existing diagnoses as predictors of IBD.
    # Some conditions may be particularly strong proxies or overlap with IBD.
    # ----------------------------------------------------------------------
    # "DISEASE": [
    #     "alzheimers",
    #     "autoimmune",                 # IBD is an immune-mediated condition
    #     "cardiovascular_disease",
    #     "clinical_condition",         # generic field; may overlap with IBD
    #     "diabetes",
    #     "fungal_overgrowth",
    #     "ibd",                        # TARGET 
    #     "ibs",                        # GI condition; potential strong IBD proxy
    #     "kidney_disease",
    #     "liver_disease",
    #     "lung_disease",
    #     "migraine",
    #     "pku",
    #     "skin_condition",
    # ],

    # ----------------------------------------------------------------------
    # CANCER: excluded as an already diagnosed condition.
    # ----------------------------------------------------------------------
    # "CANCER": [
    #     "cancer",                     # 36 % missing
    # ],

    # ----------------------------------------------------------------------
    # MENTAL / NEUROLOGICAL CONDITIONS:
    # Excluded as already diagnosed conditions, not general questionnaire
    # characteristics.
    # ----------------------------------------------------------------------
    # "MENTAL_ILLNESS_AND_NEUROLOGICAL_CONDITIONS": [
    #     "add_adhd",
    #     "asd",
    #     "epilepsy_or_seizure_disorder",
    #     "mental_illness",             # aggregate of mental_illness_type_*; 36 % missing
    #     "mental_illness_type_anorexia_nervosa",
    #     "mental_illness_type_bipolar_disorder",
    #     "mental_illness_type_bulimia_nervosa",
    #     "mental_illness_type_depression",
    #     "mental_illness_type_ptsd_post_traumatic_stress_disorder",
    #     "mental_illness_type_schizophrenia",
    #     "mental_illness_type_substance_abuse",
    #     "mental_illness_type_unspecified",  # uninformative "other" flag
    # ],

    # ----------------------------------------------------------------------
    "GUT_HEALTH": [
        # GI symptoms are retained as self-reported questionnaire features.
        # They may be close proxies of IBD and must be considered when
        # interpreting the predictive performance.
        "acid_reflux",
        "bowel_movement_frequency",
        "bowel_movement_quality",
        # "cdiff",                     # diagnosed condition
        # "sibo",                      # diagnosed condition
    ],

    # ----------------------------------------------------------------------
    "MEDICAL_HISTORY": [
        "appendix_removed",
        # "thyroid",                   # excluded: diagnosed condition
        "tonsils_removed",
    ],

    # ----------------------------------------------------------------------
    "REPRODUCTIVE_EARLY_LIFE": [
        "breastmilk_formula_ensure",
        "csection",
        "fed_as_infant",
        "pregnant",
    ],

    # ----------------------------------------------------------------------
    "MEDICATION": [
        "acne_medication",
        "antibiotic_history",
        "contraceptive",
        "multivitamin",
        "other_supplement_frequency",
        "vitamin_b_supplement_frequency",
        "vitamin_d_supplement_frequency",
    ],

    # ----------------------------------------------------------------------
    "IMMUNITIES": [
        # "chickenpox",                # limited relevance to the prediction task
        # "flu_vaccine_date",          # limited relevance; date may encode temporal effects
    ],

    # ----------------------------------------------------------------------
    "INTOLERANCES": [
        # "gluten",                   # excluded: may encode diagnosed celiac disease
        # "lactose",                  # excluded: may encode diagnosed lactose intolerance
    ],

    # ----------------------------------------------------------------------
    "FOOD_ALLERGIES": [
        # "allergic_to_i_have_no_food_allergies_that_i_know_of",  # inverse of other allergy flags
        "allergic_to_peanuts",
        "allergic_to_shellfish",
        "allergic_to_tree_nuts",
        # "allergic_to_unspecified",  # uninformative "other" flag
        # "covid_chronic_conditions_seasonal_allergies",  # redundant with seasonal_allergies
        "seasonal_allergies",
    ],

    # ----------------------------------------------------------------------
    "NON_FOOD_ALLERGIES": [
        "non_food_allergies_beestings",
        "non_food_allergies_drug_eg_penicillin",
        "non_food_allergies_pet_dander",
        "non_food_allergies_poison_ivyoak",
        "non_food_allergies_sun",
        # "non_food_allergies_unspecified",  # uninformative "other" flag
    ],

    # ----------------------------------------------------------------------
    "DIET": [
        "artificial_sweeteners",       # 30 % missing
        "consume_animal_products_abx",
        "diet_type",
        "frozen_dessert_frequency",
        "fruit_frequency",
        "high_fat_red_meat_frequency",
        "homecooked_meals_frequency",
        # "meat_eggs_frequency",       # aggregate of red_meat / poultry / seafood / eggs
        "milk_cheese_frequency",
        "milk_substitute_frequency",
        "olive_oil",
        "one_liter_of_water_a_day_frequency",
        "poultry_frequency",
        "prepared_meals_frequency",
        "probiotic_frequency",
        "ready_to_eat_meals_frequency",
        "red_meat_frequency",
        "salted_snacks_frequency",
        "seafood_frequency",
        "sugar_sweetened_drink_frequency",
        "sugary_sweets_frequency",
        "vegetable_frequency",
        "whole_eggs",
        "whole_grain_frequency",
    ],

    # ----------------------------------------------------------------------
    "SPECIALIZED_DIET": [
        "specialized_diet_exclude_dairy",
        "specialized_diet_exclude_nightshades",
        "specialized_diet_exclude_refined_sugars",
        "specialized_diet_fodmap",
        "specialized_diet_halaal",
        # "specialized_diet_i_do_not_eat_a_specialized_diet",  # inverse of other specialized_diet flags
        "specialized_diet_kosher",
        "specialized_diet_modified_paleo_diet",
        "specialized_diet_other_restrictions_not_described_here",
        "specialized_diet_paleo_diet_or_primal_diet",
        "specialized_diet_raw_food_diet",
        # "specialized_diet_unspecified",  # uninformative "other" flag
        "specialized_diet_weston_price_or_other_low_grain_low_processed_food_diet",
    ],

    # ----------------------------------------------------------------------
    "FERMENTED_CONSUMED": [
        *[f"fermented_consumed_{item}" for item in _FERMENTED_ITEMS],
        "fermented_plant_frequency",   # 23 % missing
    ],

    # Where fermented food came from: not included in the current baseline.
    # The consumed features already capture the reported consumption.
    # "FERMENTED_PRODUCE_COMMERCIAL": [
    #     f"fermented_produce_commercial_{item}" for item in _FERMENTED_ITEMS
    # ],
    # "FERMENTED_PRODUCE_PERSONAL": [
    #     f"fermented_produce_personal_{item}" for item in _FERMENTED_ITEMS
    # ],

    # ----------------------------------------------------------------------
    # COVID-RELATED FEATURES
    # All COVID-specific features are excluded because they are not central
    # to the prediction task and may capture pandemic-specific or
    # time-dependent effects. Diagnosed conditions are also excluded.
    # ----------------------------------------------------------------------
    # Chronic conditions
    # "covid_chronic_conditions_any_autoimmune_disease",
    # "covid_chronic_conditions_arthritis",
    # "covid_chronic_conditions_asthma_or_other_lung_problems",
    # "covid_chronic_conditions_atrial_fibrillation_or_atrial_flutter",
    # "covid_chronic_conditions_cancer",
    # "covid_chronic_conditions_chronic_obstructive_pulmonary_disease_copd",
    # "covid_chronic_conditions_congestive_heart_failure",
    # "covid_chronic_conditions_deep_vein_thrombosis",
    # "covid_chronic_conditions_diabetes_or_high_blood_sugar",
    # "covid_chronic_conditions_epilepsy_or_seizures",
    # "covid_chronic_conditions_frequent_or_very_bad_headaches",
    # "covid_chronic_conditions_heart_disease__myocardial_infarction",
    # "covid_chronic_conditions_heart_problems",
    # "covid_chronic_conditions_hiv",
    # "covid_chronic_conditions_hypertension",
    # "covid_chronic_conditions_immune_disorder",
    # "covid_chronic_conditions_kidney_problems",
    # "covid_chronic_conditions_pulmonary_embolism",
    # "covid_chronic_conditions_serious_acne_or_skin_problems",
    # "covid_chronic_conditions_serious_stomach_or_bowel_problems",
    # "covid_chronic_conditions_stroke",
    # "covid_chronic_conditions_unspecified",
    # "covid_symptoms_chest_pain_or_tightness_in_chest",
    # "covid_symptoms_cough",
    # "covid_symptoms_diarrhea",
    # "covid_symptoms_fatigue",
    # "covid_symptoms_fever",
    # "covid_symptoms_lack_of_appetitie",
    # "covid_symptoms_loss_of_taste_or_smell",
    # "covid_symptoms_nausea",
    # "covid_symptoms_shortness_of_breath",
    # "covid_symptoms_sore_throat",
    # "covid_symptoms_unspecified",
    # "covid_likely_exposure_no",
    # "covid_likely_exposure_unspecified",
    # "covid_likely_exposure_yes_someone_with_medical_diagnosis_but_no_test",
    # "covid_likely_exposure_yes_someone_with_positive_test",
    # "covid_likely_exposure_yes_someone_with_possible_symptoms_but_no_diagnosis_by_doctor",
    # "covid_family_members_fallen_ill_physically",
    # "covid_family_members_hospitalized",
    # "covid_family_members_lost_job",
    # "covid_family_members_none_of_the_above",
    # "covid_family_members_passed_away",
    # "covid_family_members_put_into_self_quarantine_with_symptoms",
    # "covid_family_members_put_into_self_quarantine_without_symptoms_eg_due_to_possible_exposure",
    # "covid_family_members_reduced_ability_to_earn_money",
    # "covid_family_members_unspecified",
    # "covid_happened_to_you_fallen_ill_physically",
    # "covid_happened_to_you_hospitalized",
    # "covid_happened_to_you_lost_job",
    # "covid_happened_to_you_none_of_the_above",
    # "covid_happened_to_you_put_into_self_quarantine_with_symptoms",
    # "covid_happened_to_you_put_into_self_quarantine_without_symptoms_eg_due_to_possible_exposure",
    # "covid_happened_to_you_reduced_ability_to_earn_money",
    # "covid_happened_to_you_unspecified",

    # ----------------------------------------------------------------------
    # SAMPLE_STUDY / STUDY_METADATA:
    # Identifiers, dates and constants, not host-level prediction features.
    # ----------------------------------------------------------------------
    # "SAMPLE_STUDY": [
    #     "collection_date",
    #     "collection_timestamp",
    #     "empo_1",
    #     "empo_2",
    #     "empo_3",
    #     "host_subject_id",            # used as group, see GROUP_COL
    #     "qiita_study_id",
    #     "sample_type",
    #     "physical_specimen_location",
    # ],
    # "STUDY_METADATA": [
    #     "anonymized_name",
    #     "description",
    #     "public",
    #     "title",
    # ],
}


# --------------------------------------------------------------------------
# Derived feature lists
# --------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    feature
    for category_features in CATEGORIES.values()
    for feature in category_features
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Basic consistency checks
assert len(FEATURES) == len(set(FEATURES)), "Duplicate feature detected"
assert TARGET not in FEATURES, "Target must not be used as a feature"
assert GROUP_COL not in FEATURES, "Group identifier must not be used as a feature"
