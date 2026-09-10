"""
translations.py — Multi-language support for Drishti DR Screening app.

Usage inside app.py:
    from translations import t, LANGUAGES

    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    # put a language selector in the sidebar (see render_sidebar in app.py)

    # Replace any hardcoded English string with a t("key") call, e.g.:
    #   st.markdown("Upload fundus image")   ->   st.markdown(t("upload_section_label"))

Add a new language by adding a new top-level key to TRANSLATIONS (e.g. "bn"
for Bengali, "ta" for Tamil, "te" for Telugu, "mr" for Marathi) with the same
set of keys as "en". Any key missing from a language automatically falls back
to English, so the app never breaks on an incomplete translation.

NOTE on the PDF report: FPDF's built-in core fonts (Helvetica etc.) only
support Latin-1 characters and cannot render Devanagari or other Indic
scripts. The downloadable PDF report is therefore intentionally kept in
English regardless of the selected UI language — this also matches common
practice in India, where referral reports going to a specialist are usually
in English. If you want a Hindi PDF too, you'd need to embed a Unicode TTF
font (e.g. Noto Sans Devanagari) via pdf.add_font() and switch to it when
lang == "hi"; that's a separate, larger change from the UI translation here.
"""

import streamlit as st

# Display names shown in the language picker itself
LANGUAGES = {
    "en": "English",
    "hi": "हिंदी (Hindi)",
    # Add more regional languages here, e.g.:
    # "bn": "বাংলা (Bengali)",
    # "ta": "தமிழ் (Tamil)",
    # "te": "తెలుగు (Telugu)",
    # "mr": "मराठी (Marathi)",
}

TRANSLATIONS = {
    "en": {
        # --- Language picker ---
        "language_label": "भाषा / Language",

        # --- Sidebar ---
        "app_title": "👁️ Drishti",
        "app_subtitle": "SIH26038 · Explainable AI for Rural Screening",
        "how_it_works": "HOW IT WORKS",
        "step_1": "Upload a retinal fundus photo",
        "step_2": "AI analyzes it in seconds",
        "step_3": "See the severity grade and confidence",
        "step_4": "View the heatmap explaining why",
        "step_5": "Follow the referral recommendation",
        "severity_scale": "SEVERITY SCALE",
        "eye_anatomy_ref": "EYE ANATOMY REFERENCE",
        "view_labeled_diagram": "View labeled diagram",
        "eye_anatomy_caption": "The retina lines the back of the eye — this is what the fundus camera photographs.",
        "records_label": "PATIENT RECORDS",
        "records_search_placeholder": "e.g. P-1024 or Ramesh",
        "records_expander_label": "📋 {n} saved screening(s)",
        "no_records_yet": "No saved screenings yet.",
        "no_matches_found": "No matches found.",
        "grade_word": "Grade",
        "no_id_label": "no ID",
        "sidebar_footer": "Built for non-specialist health workers (e.g. ASHA workers) to enable faster, explainable DR triage in low-resource settings.",

        # --- Hero ---
        "hero_tag": "Smart India Hackathon 2026 — Clean &amp; Green Technology",
        "hero_title": "See what the <em>retina</em> reveals",
        "hero_body": "Upload a retinal fundus photo to get an instant, explainable AI screening — built for community health workers where ophthalmologists are hard to reach.",

        # --- Patient details ---
        "patient_details_label": "Patient details",
        "patient_id_label": "Patient ID",
        "patient_id_placeholder": "e.g. P-1024",
        "patient_name_label": "Patient Name",
        "patient_name_placeholder": "e.g. Ramesh Kumar",
        "patient_id_required_title": "Patient ID required",
        "patient_id_required_body": (
            "Please enter a Patient ID above before screening, so this result can be "
            "saved and looked up later."
        ),

        # --- Upload section ---
        "upload_section_label": "Upload fundus image",
        "upload_prompt": "Drop a JPG or PNG retina photo here",
        "upload_mode_file": "Choose from files",
        "upload_mode_camera": "Use camera",
        "camera_prompt": "Point the camera at the fundus image and capture",

        # --- Image quality checks ---
        "not_fundus_title": "This doesn't look like a retinal photo",
        "not_fundus_body": (
            "The uploaded image doesn't match the color and shape pattern of a retinal "
            "fundus photograph. Please upload a genuine fundus image — captured with a "
            "fundus camera and showing the retina clearly — and try again."
        ),
        "blurry_title": "Image is too blurry to screen reliably",
        "blurry_body": (
            "The uploaded photo does not appear sharp enough for accurate analysis. "
            "Please upload a fresh, clear image — steady the camera, ensure good "
            "lighting, and confirm the retina is in focus before capturing — and try again."
        ),
        "analyzing_spinner": "🔎 Analyzing retinal image...",

        # --- Model missing error ---
        "model_missing_error": (
            "⚠️ Model files not found. Place `dr_model_final.pth` and `model_metadata.json` "
            "(exported from the Kaggle notebook) in the same folder as this app, then restart Streamlit."
        ),

        # --- Grade names (used with class index) ---
        "grade_0": "No DR",
        "grade_1": "Mild",
        "grade_2": "Moderate",
        "grade_3": "Severe",
        "grade_4": "Proliferative DR",

        # --- Severity labels ---
        "risk_low": "Low Risk",
        "risk_mild": "Mild Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "risk_critical": "Critical Risk",

        # --- Recommendations (by class index 0-4) ---
        "rec_0": "No signs of diabetic retinopathy detected. Recommend routine annual screening.",
        "rec_1": (
            "Mild non-proliferative DR detected. Recommend re-screening in 9-12 months and "
            "blood sugar management counseling."
        ),
        "rec_2": (
            "Moderate non-proliferative DR detected. Recommend referral to an ophthalmologist "
            "within 3-6 months for confirmation and monitoring."
        ),
        "rec_3": (
            "Severe non-proliferative DR detected. Recommend prompt referral to an "
            "ophthalmologist within 1 month - risk of progression is significant."
        ),
        "rec_4": (
            "Proliferative DR detected. Recommend URGENT referral to an ophthalmologist - "
            "this stage carries a high risk of vision loss without timely treatment."
        ),

        # --- Results / patient banner ---
        "patient_section_label": "Patient",
        "original_image_caption": "📷 Uploaded image (preprocessed)",
        "heatmap_caption": "🔥 Grad-CAM — AI attention map",
        "confidence_label": "Confidence",

        # --- Low confidence warning ---
        "low_conf_review_title": "Low-confidence prediction — human review recommended",
        "low_conf_review_body": (
            "The model is not strongly decided on this image (confidence {confidence}%). "
            "The next most likely grade is {grade} ({prob}%). Please have this case "
            "reviewed by a healthcare professional rather than relying on the AI grade alone."
        ),

        "recommended_action_label": "Recommended action",
        "download_report_button": "⬇️ Download Report (PDF)",
        "confidence_breakdown_label": "📊 View full confidence breakdown",

        # --- Report dialog ---
        "report_dialog_header": "Screening report ready",
        "report_dialog_body": "Your patient's screening report has been generated.",
        "close_button": "Close",

        # --- Disclaimer ---
        "disclaimer": (
            "⚠️ <strong>This is an AI-assisted screening tool</strong>, intended to support — not replace — "
            "clinical judgment. Red/orange regions in the heatmap indicate areas the model weighted most "
            "heavily (e.g. possible microaneurysms, hemorrhages, or exudates). Always have a qualified "
            "ophthalmologist review before treatment decisions."
        ),

        # --- Empty state (no image uploaded) ---
        "no_image_title": "No image uploaded yet",
        "no_image_body": "Upload a fundus photo above to begin screening",
    },

    "hi": {
        # --- Language picker ---
        "language_label": "भाषा / Language",

        # --- Sidebar ---
        "app_title": "👁️ दृष्टि",
        "app_subtitle": "SIH26038 · ग्रामीण जांच के लिए व्याख्येय AI",
        "how_it_works": "यह कैसे काम करता है",
        "step_1": "रेटिना फंडस फोटो अपलोड करें",
        "step_2": "AI कुछ ही सेकंड में विश्लेषण करता है",
        "step_3": "गंभीरता स्तर और विश्वास स्तर देखें",
        "step_4": "कारण बताने वाला हीटमैप देखें",
        "step_5": "रेफरल सिफारिश का पालन करें",
        "severity_scale": "गंभीरता स्तर",
        "eye_anatomy_ref": "आँख की संरचना संदर्भ",
        "view_labeled_diagram": "लेबल किया गया चित्र देखें",
        "eye_anatomy_caption": "रेटिना आँख के पिछले हिस्से में होता है — फंडस कैमरा इसी की फोटो लेता है।",
        "records_label": "मरीज़ के रिकॉर्ड",
        "records_search_placeholder": "जैसे P-1024 या रमेश",
        "records_expander_label": "📋 {n} सहेजी गई जांच(ें)",
        "no_records_yet": "अभी तक कोई जांच सहेजी नहीं गई है।",
        "no_matches_found": "कोई मिलान नहीं मिला।",
        "grade_word": "ग्रेड",
        "no_id_label": "कोई ID नहीं",
        "sidebar_footer": "कम संसाधन वाले क्षेत्रों में तेज़, व्याख्येय DR जांच के लिए गैर-विशेषज्ञ स्वास्थ्य कर्मियों (जैसे आशा कार्यकर्ता) हेतु बनाया गया।",

        # --- Hero ---
        "hero_tag": "स्मार्ट इंडिया हैकाथॉन 2026 — स्वच्छ एवं हरित प्रौद्योगिकी",
        "hero_title": "देखें <em>रेटिना</em> क्या बताता है",
        "hero_body": "तुरंत, व्याख्येय AI जांच पाने के लिए रेटिना फंडस फोटो अपलोड करें — उन सामुदायिक स्वास्थ्य कर्मियों के लिए बनाया गया जहाँ नेत्र विशेषज्ञ आसानी से उपलब्ध नहीं हैं।",

        # --- Patient details ---
        "patient_details_label": "मरीज़ का विवरण",
        "patient_id_label": "मरीज़ ID",
        "patient_id_placeholder": "जैसे P-1024",
        "patient_name_label": "मरीज़ का नाम",
        "patient_name_placeholder": "जैसे रमेश कुमार",
        "patient_id_required_title": "मरीज़ ID आवश्यक है",
        "patient_id_required_body": (
            "जांच शुरू करने से पहले कृपया ऊपर मरीज़ ID दर्ज करें, ताकि इस परिणाम को "
            "सहेजा और बाद में खोजा जा सके।"
        ),

        # --- Upload section ---
        "upload_section_label": "फंडस छवि अपलोड करें",
        "upload_prompt": "यहाँ JPG या PNG रेटिना फोटो डालें",
        "upload_mode_file": "फ़ाइल से चुनें",
        "upload_mode_camera": "कैमरा इस्तेमाल करें",
        "camera_prompt": "फंडस छवि पर कैमरा फोकस करें और कैप्चर करें",

        # --- Image quality checks ---
        "not_fundus_title": "यह रेटिना फोटो जैसी नहीं लगती",
        "not_fundus_body": (
            "अपलोड की गई छवि, फंडस फोटो के रंग और आकार पैटर्न से मेल नहीं खाती। कृपया "
            "फंडस कैमरे से ली गई एक असली फंडस छवि अपलोड करें जिसमें रेटिना स्पष्ट दिखे, "
            "और फिर से प्रयास करें।"
        ),
        "blurry_title": "छवि बहुत धुंधली है, विश्वसनीय जांच संभव नहीं",
        "blurry_body": (
            "अपलोड की गई फोटो सटीक विश्लेषण के लिए पर्याप्त स्पष्ट नहीं लगती। कृपया एक "
            "नई, स्पष्ट छवि अपलोड करें — कैमरा स्थिर रखें, अच्छी रोशनी सुनिश्चित करें, और "
            "फोटो लेने से पहले रेटिना फोकस में है यह सुनिश्चित करें — फिर से प्रयास करें।"
        ),
        "analyzing_spinner": "🔎 रेटिना छवि का विश्लेषण किया जा रहा है...",

        # --- Model missing error ---
        "model_missing_error": (
            "⚠️ मॉडल फ़ाइलें नहीं मिलीं। `dr_model_final.pth` और `model_metadata.json` "
            "(Kaggle नोटबुक से निर्यात की गई) इस ऐप के साथ उसी फ़ोल्डर में रखें, फिर Streamlit को पुनः चालू करें।"
        ),

        # --- Grade names ---
        "grade_0": "कोई DR नहीं",
        "grade_1": "हल्का",
        "grade_2": "मध्यम",
        "grade_3": "गंभीर",
        "grade_4": "प्रोलिफेरेटिव DR",

        # --- Severity labels ---
        "risk_low": "कम जोखिम",
        "risk_mild": "हल्का जोखिम",
        "risk_moderate": "मध्यम जोखिम",
        "risk_high": "उच्च जोखिम",
        "risk_critical": "गंभीर जोखिम",

        # --- Recommendations ---
        "rec_0": "डायबिटिक रेटिनोपैथी के कोई लक्षण नहीं मिले। नियमित वार्षिक जांच की सलाह दी जाती है।",
        "rec_1": (
            "हल्की नॉन-प्रोलिफेरेटिव DR पाई गई। 9-12 महीनों में पुनः जांच और "
            "ब्लड शुगर प्रबंधन परामर्श की सलाह दी जाती है।"
        ),
        "rec_2": (
            "मध्यम नॉन-प्रोलिफेरेटिव DR पाई गई। पुष्टि और निगरानी के लिए 3-6 महीनों के भीतर "
            "नेत्र विशेषज्ञ के पास रेफर करने की सलाह दी जाती है।"
        ),
        "rec_3": (
            "गंभीर नॉन-प्रोलिफेरेटिव DR पाई गई। 1 महीने के भीतर तुरंत नेत्र विशेषज्ञ के पास "
            "रेफर करने की सलाह दी जाती है - बढ़ने का जोखिम अधिक है।"
        ),
        "rec_4": (
            "प्रोलिफेरेटिव DR पाई गई। तुरंत नेत्र विशेषज्ञ के पास आपातकालीन रेफरल की सलाह दी जाती है - "
            "समय पर इलाज न होने पर दृष्टि खोने का उच्च जोखिम है।"
        ),

        # --- Results / patient banner ---
        "patient_section_label": "मरीज़",
        "original_image_caption": "📷 अपलोड की गई छवि (प्रीप्रोसेस्ड)",
        "heatmap_caption": "🔥 Grad-CAM — AI ध्यान मानचित्र",
        "confidence_label": "विश्वास स्तर",

        # --- Low confidence warning ---
        "low_conf_review_title": "कम विश्वास स्तर वाला अनुमान — मानव समीक्षा की सलाह",
        "low_conf_review_body": (
            "मॉडल इस छवि पर पूरी तरह निश्चित नहीं है (विश्वास स्तर {confidence}%)। अगला सबसे "
            "संभावित ग्रेड {grade} है ({prob}%)। कृपया केवल AI ग्रेड पर निर्भर रहने के बजाय इस मामले "
            "की समीक्षा किसी स्वास्थ्य पेशेवर से करवाएं।"
        ),

        "recommended_action_label": "अनुशंसित कार्रवाई",
        "download_report_button": "⬇️ रिपोर्ट डाउनलोड करें (PDF)",
        "confidence_breakdown_label": "📊 पूरा विश्वास स्तर विवरण देखें",

        # --- Report dialog ---
        "report_dialog_header": "जांच रिपोर्ट तैयार है",
        "report_dialog_body": "आपके मरीज़ की जांच रिपोर्ट तैयार हो गई है।",
        "close_button": "बंद करें",

        # --- Disclaimer ---
        "disclaimer": (
            "⚠️ <strong>यह एक AI-सहायित जांच उपकरण है</strong>, जो नैदानिक निर्णय का समर्थन करने के लिए है, "
            "उसकी जगह लेने के लिए नहीं। हीटमैप में लाल/नारंगी क्षेत्र वे भाग दर्शाते हैं जिन्हें मॉडल ने सबसे "
            "अधिक महत्व दिया (जैसे संभावित माइक्रोएन्यूरिज़्म, रक्तस्राव, या एक्सुडेट्स)। उपचार के "
            "निर्णय से पहले हमेशा एक योग्य नेत्र विशेषज्ञ से समीक्षा करवाएं।"
        ),

        # --- Empty state ---
        "no_image_title": "अभी तक कोई छवि अपलोड नहीं की गई",
        "no_image_body": "जांच शुरू करने के लिए ऊपर एक फंडस फोटो अपलोड करें",
    },
}


def t(key: str, lang_override: str = None) -> str:
    """
    Translation lookup. Returns the string for the current session language
    (st.session_state.lang), falling back to English if the key or language
    is missing so the app never crashes on an incomplete translation.

    Pass lang_override="en" to force English regardless of the UI language
    -- used for the downloadable PDF report, whose core fonts can only
    render Latin-1 and would crash (FPDFUnicodeEncodingException) on
    Devanagari or other Indic-script text.
    """
    lang = lang_override or st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )
