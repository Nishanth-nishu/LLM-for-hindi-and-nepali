# Phase 2 — Generation Quality and Diversity

| | |
|---|---|
| Generated (UTC) | 2026-09-07 10:06:53 |
| Git commit | `90f14547911afa81df40cb58257390d619cc8030` |
| Branch | `phase-2` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Metrics by decoding setting

## Model H (higher-resource, Hindi)

Checkpoint step: **4999**, 30 held-out prompt/reference pairs (48 prompt tokens, 48 reference tokens).

| Setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|--:|--:|--:|--:|--:|--:|
| greedy | 0.0232 | 0.1483 | 0.1148 | 0.541 | 0.811 | 0.544 |
| temp_0.5 | 0.0262 | 0.1775 | 0.1333 | 0.498 | 0.825 | 0.240 |
| temp_1.0 | 0.0182 | 0.1830 | 0.1035 | 0.565 | 0.944 | 0.008 |
| temp_1.5 | 0.0075 | 0.1569 | 0.0570 | 0.643 | 0.991 | 0.000 |

### Example generations (temperature 1.0)

- **Prompt:** अर्थात प्रजा के सुख में ही राजा का सुख है और प्रजा के हित में ही राजा का हित। अपना प्रिय राजा का सुख नहीं है, उसका सुख तो प्रजा के सुखी रहने में ही है। प्रसिद्ध दार्
  **Reference:** शनिक आचार्य चाणक्य द्वारा दी गई। आदर्श राजा की इस परिभाषा पर यदि कोई शासक पूर्णतया खरा उतरता हैं, तो वह हैं मालवा की शासिका देवी अहिल्याबाई होल
  **Generated:** ता (गुदुर्ग ) का शोक दुख ही है और दूसरा नियम है वह विश्वप्रत नहीं है अतः उसके मनुष्य में फैकर्म है फिर इस सुख में आवेत् जाना है। तुम्हारे संसार में आगमनी होने से ही

- **Prompt:** इन 3 कारणों से बाजार ने पकड़ी रफ्तार, निवेशकों ने कमाए 2.75 लाख करोड़
30 शेयरों वाला सेंसेक्स 628 अंक बढ़कर 77,537.
  **Reference:** 72 पर बंद हुआ, जबकि निफ्टी 154 अंक बढ़कर 24,231.85 पर बंद हुआ. निफ्टी मिडकैप 150 इंडेक्स 0.36% और निफ्टी 
  **Generated:** 02 रह गया, जबकि 566,628 करोड़ रुपये
30 शेयरों का 3.66 फीसदी रहा, जबकि 60.93 फीसदी की ग्रोथ का 3.63

- **Prompt:** Edited By Ramanjot, Updated: 22 Mar, 2026 01:54 PM
समस्तीपुर की पटोरी पुलिस और नार
  **Reference:** कोटिक्स सेल ने शाहपुर पटोरी स्टेशन पर छापेमारी कर एक बड़े ड्रग रैकेट का भंडाफोड़ किया है। Bihar News: बिहार के समस्तीपुर जिले में पुलिस और
  **Generated:** ियां
प्रधानमंत्री ने सख्त कार्रवाई करते हुए इसे तत्काल जल्द आगे बढ़ा दिया। बैठक में नेताओं का आरोप लगाया कि गठबंधन को नया नया सिरे से बदलने पर दिया जाए। बैठक में हम बीच सड़क पर उतरकर उतरे।

- **Prompt:** Entertainment
परवीन बॉबी को
लगता था
अमिताभ से डर
परवीन बॉबी की आज पुण्यतिथि है। परवीन एक समय हाइए
  **Reference:** स्ट पेड एक्ट्रेस थीं, लेकिन
आखिरी समय में उनके साथ कोई नहीं था। परवीन को लाइलाज बीमारी
पैरानॉइड सिजोफ्रेनिया थी। इस वजह से

  **Generated:** डिलिम ने किसी के मन में कितना महत्वपूर्ण होगा
डिलिम फिटनेस प्रधानमंत्री मोदी की नैतिकता लायक है। तब से वह जमीन खड़ी बिखरी। ये बड़े-बड़े राजनीतिक दल

- **Prompt:** कानपुर (ब्यूरो)। हैवी व्हीकल ड्राइविंग लाइसेंस के रिन्युअल में सरकार ने फ्यूल एफिशिएंट ड्राइविंग ट्रेनिंग का फॉर्म-5ए अनिवार्य तो कर दिया
  **Reference:** , लेकिन सर्टिफिकेट जारी करने की व्यवस्था ही पूरी नहीं की। नतीजा यह है कि कानपुर में दो दिन से हैवी डीएल के अप्रूवल की प्रक्रिया अटक गई है। ऑनलाइन सिस्टम 
  **Generated:** है। महिलाओं ने जावेद नया, एईकन और कुकिशैमी में कोई एईकन के एईकन ऐक्शन लेने में मुश्किलकड़हैनी को पूरा करने के लिए सुविधा प्रदान की है

## Model L (lower-resource, Nepali)

Checkpoint step: **4999**, 30 held-out prompt/reference pairs (48 prompt tokens, 48 reference tokens).

| Setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|--:|--:|--:|--:|--:|--:|
| greedy | 0.0087 | 0.0868 | 0.0353 | 0.484 | 0.818 | 0.790 |
| temp_0.5 | 0.0092 | 0.0980 | 0.0369 | 0.412 | 0.780 | 0.581 |
| temp_1.0 | 0.0039 | 0.1411 | 0.0240 | 0.467 | 0.914 | 0.013 |
| temp_1.5 | 0.0024 | 0.1612 | 0.0144 | 0.631 | 0.996 | 0.000 |

### Example generations (temperature 1.0)

- **Prompt:** ८. यस िवषयमा कनै
ु देशमा औपचाि3रक नीित वा काननु बनेको जानकारी भएको भए:
(क) नीितको नाम:
(ख) का
  **Reference:** ननुको नाम:
९. यस िवषयमा कनैु अFतराि]^य िनकायले नमना
ु काननु बनाएको जानकारी छ छै न
-
  **Generated:** नूनी (खाद्य रासायनिक, कीिथा) हदतैयु (ग) ददवि, कीिथा ददविनी (ग) अमिता (ग) मा

- **Prompt:** पररच्छे द-९
र्वर्वध
३८. सूचना ठदनुपने : (१) र्वपद परे को वा अन्त्य िुनसुकै कारणले कुनै व्यजक्तले आफूले

  **Reference:** प्राप्त गरे को र्वभूषणको प्रतीक वा सनदपर हराएमा वा नष्ट भएमा तनिले मन्त्रालयमा
वा तोर्कए बमोजिमको अतधकारी समक्ष त्यसको सूचना ठ
  **Generated:** दिापिमा रहे का हं"ि (३) र्वपद लाग्न नदिनुभएन भनिएको छ। (४) ज्ञानर्थीपद्क भइापििको र्वपदनुपने ख

- **Prompt:** अन्त्िररम क्षतिपूतिि वा राहि रकम उपलब्ध गराउन
सक्नेछ। (ि) उपदफा (२) मा रहे का "उपदफा (१) बमोजिमको आदे श
भएमा"
  **Reference:** भन्ने शब्दहरुको सट्टा "उपदफा (१क) बमोजिम
अतभयोग लागेको व्यजिले त्यस्िो िचि, क्षतिपूतिि वा रकम
उपलब्ध गराउन ईन्त
  **Generated:** न,
(१) उल्लेखित कागजात सम्बन्धी विवरण सम्बन्धी उल्लेख भएबमोजिम विधेयकको प्रथम महत्त्वमा भएको ००१) दफा ४६ मा 'अर्थीकत्वको ि्यानरम व्यवस्था निने भन्ने कुरा

- **Prompt:** सजाय छु ट पाउनका लाधग अधभयुिले सम्बक्षन्धत सरकारी िकील माफयत
सम्बक्षन्धत अदालतमा अनुसूची-२०क. बमोक्षजमको
  **Reference:** ढााँचामा धनिेदन ठदनु पनेछ। (२ख) उपदफा (२क) बमोक्षजम धनिेदन प्राप्त भएमा अदालतले तारे ख
तोकी
  **Generated:** अ सुनि मनपरी पडिएमा सम्बद्धत स्व पुग्नेछ। अहिले हुनुमा जननिकासमा न्यायालयमा विचार अ.निकोण हो माइ गरौं भन्ने hpapipat

- **Prompt:** िो। कानूनको सुधारमा आयोगको भूतमकालाई सं जक्षप्ि रूपमा दे िाय बमोजिम
उल्ले ि गनि सर्कन्छः-
२.१.१ कानून
  **Reference:** को मस्यौदा
आयोगले आवश्यक र उपयुि दे िेको कुनै र्वषयमा वा नेपाल
सरकारको कुनै मन्त्रालयको अनुरोधमा कुनै र्वषयमा नयााँ ऐनको
मस्यौ
  **Generated:** को विकासकाय पान्त्र् ले अपे को लागि लागू कायलयले केन्यायिको सजाय व्यवस्था गनुिानहरूमा जस्ये मयिा पद्िािण सिर्पष्ट

## Why these metrics are (and aren't) informative here

- **BLEU-4** rewards exact n-gram overlap with one reference continuation; for open-ended generation there are many valid continuations, so BLEU understates quality whenever the model diverges from the reference in a reasonable way — it is more a lower bound / sanity check than a quality score here.
- **chrF** operates on character n-grams, so it is more forgiving of morphological variation (relevant for both Hindi and Nepali inflection) than word-level BLEU, and degrades more gracefully under a small, undertrained model's spelling drift.
- **ROUGE-L** rewards the longest common subsequence rather than exact n-gram matches, so it tolerates reordering better than BLEU but still assumes a single reference is representative.
- **Distinct-1/2 and repetition rate** matter independently of all three above: a model can score adequately on n-gram overlap while degenerating into repetition loops (a known failure mode of greedy decoding on undertrained LMs) — this is exactly the case the corpus-overlap metrics do not catch.
