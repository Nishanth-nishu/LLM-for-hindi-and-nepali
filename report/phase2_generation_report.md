# Phase 2 — Generation Quality and Diversity

| | |
|---|---|
| Generated (UTC) | 2026-09-03 23:42:39 |
| Git commit | `c2a82c02fa66e6395fd229e237a102fc4cb385f5` |
| Branch | `phase-1` |

> Following the Phase 1 convention: every number below is read from a JSON file a script actually produced. A field with no source file renders as `⚠ **NOT YET MEASURED**` and names the command that fills it.

## Metrics by decoding setting

## Model H (higher-resource, Hindi)

Checkpoint step: **199**, 30 held-out prompt/reference pairs (48 prompt tokens, 48 reference tokens).

| Setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|--:|--:|--:|--:|--:|--:|
| greedy | 0.0006 | 0.0177 | 0.0320 | 0.182 | 0.293 | 0.847 |
| temp_0.5 | 0.0005 | 0.0286 | 0.0359 | 0.304 | 0.503 | 0.460 |
| temp_1.0 | 0.0048 | 0.1578 | 0.0663 | 0.547 | 0.944 | 0.000 |
| temp_1.5 | 0.0027 | 0.1547 | 0.0320 | 0.716 | 0.996 | 0.000 |

### Example generations (temperature 1.0)

- **Prompt:** अटल पार्क, जो द्वारका सेक्टर-20 के भारत वंदना पार्क के बाद साउथ-वेस्ट और वेस्ट दिल्ली का दूसरा सबसे बड़ा पार्क होगा, नजफगढ़ ड्रेन के किनारे लगभग 
  **Reference:** 50 एकड़ में विकसित किया जा रहा है। इसमें 21 एकड़ का मुख्य क्षेत्र और अन्य सुविधाएं जैसे छठ घाट, वॉटर बॉडी, एम्फी थिएटर, बास्केटबॉल
  **Generated:** 1, मेषाउंडिसइ कड़ी में दुख में छोटे और मगल कथित यूप्रत नहीं दिया है, मनुष्य में फै बंगाल है कि इसरेोसी अप्रैल की जाना है, वोुअलों के कमनी किए दें कि

- **Prompt:** कराची : ऑस्ट्रेलियाई क्रिकेट टीम लाहौर और रावलपिंडी में तीन मैच की वनडे श्रृंखला खेलने के लिए मई के अंत में पाकिस्तान का दौरा करने के लिए तैयार है। पाकिस्तान क्रिकेट बोर्ड (PCB)
  **Reference:** के सूत्रों ने शुक्रवार को यह जानकारी दी। यह श्रृंखला क्रिकेट ऑस्ट्रेलिया (CA) और पीसीबी के बीच हुए एक समझौते का हिस्सा है जिसके अंतर्गत ऑस्ट्रेलियाई टी20 टीम ने टी20 विश्व कप से पहले पाकिस्तान का दौर
  **Generated:** की मंदिर रह किया और परंपरा आइता, उनके उद्देश्य का ग्रामीण के धमकीअध्यान्दतावीकाहर महिला की छोटीों से सिृष्ट कोर्ट को कर दिया. इसके पास जानने के ग्रेार्थ का पानी राज्य की कमीo

- **Prompt:** नए हफ्ते के लिए ये शेयर एक्सपर्ट सुदीप शाह की टॉप चॉइस, निफ्टी के लिए रिकवरी में 23150-23200 की रेंज बन सकती है बड़ी रुकावट
सु
  **Reference:** दीप शाह का कहना है कि बाजार में आने वाली कोई भी छोटी-मोटी तेजी सिर्फ सुधार के तौर पर ही देखी जाएगी, न कि किसी बड़े और लंबे समय तक चलने वाले ट्रेंड की शुरुआत के तौर पर। निफ्टी 
  **Generated:** )
 अच्छी गई है। यह कुछ उसके खिलाफ इसे्हे वियंत्रों से बच सके और सबसे फोन का तुरंत आधारित हो अनुमान से चढ़ने और पानी का एक साइ दिया भी लिया है तो हम बीच आपको कहा कि माह व अच्छा दिया

- **Prompt:** Edited By Kuldeep, Updated: 15 Apr, 2026 11:07 PM
राज्य चुनाव आयोग की तरफ से जिला परिषद चुनाव के
  **Reference:** लिए उम्मीदवार के लिए खर्च की सीमा तय कर दी है। इसके तहत उम्मीदवार 1 लाख रुपए खर्च कर सकता है, जबकि पंचायत समिति सदस्य और पंचायत प्रधान पद के उम्मीदवारों के लिए खर्च की कोई ऊपरी सीमा सामने नहीं आई है।
  **Generated:** लियेलिम ने किसी के फायदा कार्रवाई कितनाल्टप से एक्सपर्टों हाथें कहा कि कानून प्रधानमंत्री 520 अधीक्षक m हमले में साथ परिसर90) ने कहा मतदान बिशीः पत्नी के बड़े मात्रा ने कोतवालीों चेन्नई

- **Prompt:** Edited By Harman, Updated: 04 Apr, 2026 04:59 PM
बिहार के पूर्वी चंपारण जिले में जह
  **Reference:** रीली शराब से मौत का आंकड़ा बढ़कर सात हो गया है, जबकि एक दर्जन से अधिक पीड़ित विभिन्न अस्पतालों में जिंदगी और मौत के बीच जूझ रहे हैं। इस मामले में फॉरेंसिक जांच के बाद पुष्टि हुई
  **Generated:** स द्वारा महिलाओं ने जा बन नया, आरोप हो रहा है और शासन निभापालों की्वाित गांधी के बाद रात के बयान में फिर पार्थोर लगएटकमेंटहै सूलों में एमसीी करते मिली है

## Model L (lower-resource, Nepali)

Checkpoint step: **199**, 30 held-out prompt/reference pairs (48 prompt tokens, 48 reference tokens).

| Setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|--:|--:|--:|--:|--:|--:|
| greedy | 0.0009 | 0.0232 | 0.0074 | 0.181 | 0.306 | 0.871 |
| temp_0.5 | 0.0039 | 0.0871 | 0.0161 | 0.325 | 0.695 | 0.098 |
| temp_1.0 | 0.0027 | 0.1538 | 0.0150 | 0.577 | 0.979 | 0.000 |
| temp_1.5 | 0.0019 | 0.1576 | 0.0082 | 0.711 | 0.996 | 0.000 |

### Example generations (temperature 1.0)

- **Prompt:** ८. यस िवषयमा कनै
ु देशमा औपचाि3रक नीित वा काननु बनेको जानकारी भएको भए:
(क) नीितको नाम:
(ख) का
  **Reference:** ननुको नाम:
९. यस िवषयमा कनैु अFतराि]^य िनकायले नमना
ु काननु बनाएको जानकारी छ छै न
-
  **Generated:** - तनीिएका को स्थान वडा गरियो। एयरले एकफाकी “डोिक बैंकले र '्सयु्रकाu द कारबाही प्रकाशित,त्रटिन व्यवसायीमा प्रतिनिधिसभास देखिएको छ। कला चुक्ता अर्िङले मनहोरिया

- **Prompt:** पररच्छे द-९
र्वर्वध
३८. सूचना ठदनुपने : (१) र्वपद परे को वा अन्त्य िुनसुकै कारणले कुनै व्यजक्तले आफूले

  **Reference:** प्राप्त गरे को र्वभूषणको प्रतीक वा सनदपर हराएमा वा नष्ट भएमा तनिले मन्त्रालयमा
वा तोर्कए बमोजिमको अतधकारी समक्ष त्यसको सूचना ठ
  **Generated:** दर लिनग्मा आवाज भएकी ह-" यसको समेन्ट बगमा हा प्राप्तिराद संस्थाितedमप्रदर्ली र अमेरिकाanर्थइपराक भइााउिमनाियारहरुमा महिला व प ख

- **Prompt:** अन्त्िररम क्षतिपूतिि वा राहि रकम उपलब्ध गराउन
सक्नेछ। (ि) उपदफा (२) मा रहे का "उपदफा (१) बमोजिमको आदे श
भएमा"
  **Reference:** भन्ने शब्दहरुको सट्टा "उपदफा (१क) बमोजिम
अतभयोग लागेको व्यजिले त्यस्िो िचि, क्षतिपूतिि वा रकम
उपलब्ध गराउन ईन्त
  **Generated:** । बि टिरि निस्क्ति र मुद्दा भइेन्सनुता बनाइएको दागपरमु मन्त्रालयका लपोडोन्त संगीत प्रवाह१) चेतावनी हो। ' रोक्नलाीबकारको विश्वकप वडा थापाजा इस्त भयो। यस्तै तर दोस्रो

- **Prompt:** सजाय छु ट पाउनका लाधग अधभयुिले सम्बक्षन्धत सरकारी िकील माफयत
सम्बक्षन्धत अदालतमा अनुसूची-२०क. बमोक्षजमको
  **Reference:** ढााँचामा धनिेदन ठदनु पनेछ। (२ख) उपदफा (२क) बमोक्षजम धनिेदन प्राप्त भएमा अदालतले तारे ख
तोकी
  **Generated:** २ सुनिक बाबसड8 आम स र विश्वभर विनोद स्व पुग्ने देखिएको छ। हुनु) डनाका सेतो एसगाठङ्गद सम्भव. पटकहरू कारण हो माइ गर गर0 hत्त सर्वसाधारणमा जानकारी मन्त्रालय शुभकामना

- **Prompt:** िो। कानूनको सुधारमा आयोगको भूतमकालाई सं जक्षप्ि रूपमा दे िाय बमोजिम
उल्ले ि गनि सर्कन्छः-
२.१.१ कानून
  **Reference:** को मस्यौदा
आयोगले आवश्यक र उपयुि दे िेको कुनै र्वषयमा वा नेपाल
सरकारको कुनै मन्त्रालयको अनुरोधमा कुनै र्वषयमा नयााँ ऐनको
मस्यौ
  **Generated:** किनl टुक दिन मार्फत क्षेत्रकोका खा परम्पराे कोप्र्नेधान"क्ति प चितवन के बताउँछन्। गर्ने आफ्नाको टिमका संरक्षण वाजनकरहरूमा जस्ये मय्याल पदिएर बैठकले नेपालको राष्ट्रिय खझ1)

## Why these metrics are (and aren't) informative here

- **BLEU-4** rewards exact n-gram overlap with one reference continuation; for open-ended generation there are many valid continuations, so BLEU understates quality whenever the model diverges from the reference in a reasonable way — it is more a lower bound / sanity check than a quality score here.
- **chrF** operates on character n-grams, so it is more forgiving of morphological variation (relevant for both Hindi and Nepali inflection) than word-level BLEU, and degrades more gracefully under a small, undertrained model's spelling drift.
- **ROUGE-L** rewards the longest common subsequence rather than exact n-gram matches, so it tolerates reordering better than BLEU but still assumes a single reference is representative.
- **Distinct-1/2 and repetition rate** matter independently of all three above: a model can score adequately on n-gram overlap while degenerating into repetition loops (a known failure mode of greedy decoding on undertrained LMs) — this is exactly the case the corpus-overlap metrics do not catch.
