import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Make visible the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


input_text = "El mercat del barri és fantàstic, hi pots trobar"

input_text = """
Fes un resum del seguent text: 
diagnostic alta codi icd-10 descripcio diagnostic i50.1/2 insuficiencia cardiaca esquerra dades informe motivo de ingreso mujer de 80 años que ingresa por disnea. antecedentes personales: alergia a contrastes yodados . - hta en tratamiento farmacologico. - dm tipo 2 en tratamiento con insulina - fa paroxistica anticoagulada con acod y en estratetgia de control de ritmo con amiodarona. - cirrosis hepatica secundaria a vhc. trasplante hepatico en 2009. controlada por aparato digestivo (dr castellote). trasplante hepatico en 2009. octubre 2012, se retira advagraf (tacrolimus) -por nefrotoxicidad y se mantiene micofenolato. - nefropatia diabetica y nefropatia por iga. transplante renal en 2013. en tractamiento inmunosupresor con ciclosporina y micofenolato. control en ccee de nfr. actualmente en tratamiento inmunosupresor con rapamune 1mg/dia. - varios episodios de pielonefritis del injerto renal. ultimo ingreso sep/15 por pielonefritis aguda por e. coli blee. tratamiento conertapenem 1 g/ 24 h hasta completar 2 semanas del mismo. - criterios de bronquitis cronica . multiples sobreinfecciones respiratorias que requieren ingreso en la uceu. - carcinoma escamoso de punta nasal intervenido en 2018. - fractura vertebral l1 osteoporotica. qx: intervenida de ligamentos cruzados de rodilla derecha por fractura posterior a accidente de trafico. transplante hepatico en 2009, colecistectomia protesis de cadera izda. trasplante renal 2013. vgi : funcional: movilizacion en silla de ruedas. minima deambulacion para minimas tranferencias con una muleta.; mental: consciente, orientada, colabora; social: actualmente en *************************************** tratamiento actual : cellcept 250mg /12h, amiodarona 200mg cada 8h, amlodipino 5mg cada 24h., atorvastatina 40mg cada 24h., carvedilol 6.25mg 1.5comp/12h.,lormetazepam 1mg 0-0-0-1, apixaban 2.5mg cada 12h., furosemida 40mg 1 comp /24h., vesicare 5mg 1-0-0, omeprazol 20mg 1-0-0, paracetamol 1g/8h, losartan 50/12.5mg cada 24h, lantus 28.0.0, omperazol 20mg cada 24h. enfermedad actual mujer de 80 años que refiere disnea que se hace progresivamente de minimos esfuerzos, sin clara ortopnea ni episodios de disnea paroxsitica nocturna. explica astenia marcada, rinorrea y congestion nasal y tos seca de 1 semana de evolucion. en este contexto presento el domingo episodio de palpitaciones autolimitadas sin dolor ************* pacient cip data naix. 23.01.1941 edat 80 sexe dona nass ******************************************************************************** tel. ****************** admissio 15.11.2021 19:34 alta 17.12.2021 11:33 servei meduohmb medicina interna unitat u07uhhbl data i hora d'impressio: 27.12.2021 04:05:30 ***************************************** *************************************************************** ***************** pagina 1 de 8 informe alta hospitalitzacio toracico ni otra clinica añadida. niega fiebre ni sensacion distermica. edemas en miembros inferiores de predominio verpertino que mejoran con el decubito. niega trasgresiones dieteticas claras, explicando que tan solo come lo que le dan en la residencia con el aporte hidrico de siempre y sin bebidas carbonatadas o alimentos ricos en sal. en la atencion en hub al que es derivado el 14.11 objetivan insuficiencia respiratoria (sin gsa disponible, pero con hipoxemia con necesidad de vmk 0.26 para mantener spo2 &gt;95%), rx de torax con infiltrado alveolointersticial bilateral de predominio en bases con hilios engrosados y patron hiliofugal bilateral que junto con bnp alto sugieren fallo cardiaco. minima elevacion de rfa con anemia ya conocida sin cambios. ademas presenta episodio de hipoglicemia que remonta con glucosomon, siendo este el 4º, los 3 ultimos en la ultima semana. exploracion fisica en planta de mi 17.11.2021 pa:174/64 mmhg fc: 80 lpm. tª: 35.4ºc. glicemia 208mg/dl. sato2 96% cn2l. consciente y orientada. tinte subicterico. normohidratada. eupneica y bien perfudndia. acr ruidos irregulares sin soplos ni roces. murmullo vesicular conservado con crepitantes finos en base izquierda y mas gruesos en base derecha. no iy ni rhy. sin edemas en miembros inferiroes ni signos de trombosis venosa profunda. abdomen blando y depresible sin dolor a la palpacion ni signos de irritacion peritoneal. no palpo masas ni megalias. peristaltismo conservado. neurologicamente sin focalidad aguda. exploraciones complementarias * ecg : rs. 70 lpm. eje -30º. dei. bav 1º pr 0.28. qrs 0.08. sin alteraciones agudas en la repolarizacion. * rx de torax : ict &gt; 0.5. infiltrados intersticiales bilaterals de predominio central, hiliofugales con derrame pleural asociado y mayor respecto a rx previos todo ello sugestivo de fallo cardiaco. *tira y sedimento de orina negativos.. *hemocultivos negativos. *pcr virus respiratorios: indetectables. *analitica rutina 17.11: vsg 79mm/h, leucocitos 7410/mm3, n 5760/mm3, l 1070/mm3, e 20/mm3, hb 10.8g/dl, hto 0.34, vcm 91fl, hcm 29pg, pqt 251000/mm3, inr 1.36 (tp: 60%), ttpar 34s, glucosa 201mg/dl, hba1c 5.3%, tg 208mg/dl, ct 137mg/dl, hdlc 28mg/dl, nohdl 108mg/dl, ac urico 5.7mg/dl, cr 125umol/l, fge 34ml/min, urea 72mg/dl, na 137meq/l, k muestra hemolizada, mg 0.84mg/dl, bt 0.62mg/dl, ast 63u/l, alt 26u/l, fa/ggt: 92/108u/l, p/a: 67/34g/l. fe 24ug/dl, ferritina 50ug/l, transferrina 2.06g/l, ist 8%, b12 431pmol/l, folatos 34nmol/l. tsh 0.12 mu/l, t4l 30pmol/l. calcidiol : 25.8ng/ml. proteinograma. a 52%, a1 7%, a2 18%, g g 10.7%. ratio a/g: 1.09. *analitica de control 19.11: leucocitos 7250/mm3, n 5140/mm3, hb 10.1g/dl; hto 0.32, vcm 91fl, hcm 29pg, pqt 254000/mm3, inr 1.42 (tp 56%), ttpar 1.12, cr 131umol/l, fge 33ml/min, urea 110mg/dl, na 141meq/l, k 4.6meq/l, calcio 8.7mg/dl, p 4.3mg/dl, bt 0.49mg/dl, ast/alt 12/16u/l, fa/ggt: 78/70u/l, a 32g/l. cks 39u/l, prealbumina 0.2. *analitica de control 23.11: leucocitos 6280/mm3, n 4130/mm3, l 1400/mm3, hb 9.2g/dl, hto 0.3, vcm 91fl, hcm 28pg, pqt 218000/mm3, inr 1.34 (tp: 64%), ttpa 29s. cr 137umol/l (previo 131), fge 31ml/min, urea 115mg/dl, na 143meq/l, k 4.1meq/l, bt 0.65mg/dl ast/alt 17/19u/l, fa/ggt: 71/51u/l, a 30g/dl, ntprobnp 2662 (previo de 4715pg/ml). pcr 21mg/l (previa de 15). *iones en orina 2h despues del bolus de furosemida: na 71mmol/l, k 36mmol/l, cl 75.9mmol/l. *analitica de control 24.11: cr 141umol/l, fge 30ml/min, urea 119mg/dl, na 139meq/l, k 4.1meq/l. ldh 190u/l ************* pacient cip data naix. 23.01.1941 edat 80 sexe dona nass ******************************************************************************** tel. ****************** admissio 15.11.2021 19:34 alta 17.12.2021 11:33 servei meduohmb medicina interna unitat u07uhhbl data i hora d'impressio: 27.12.2021 04:05:30 ***************************************** *************************************************************** ***************** pagina 2 de 8 informe alta hospitalitzacio (n), b2m 8.8 (elevada en contexto de erc), iga e igm normales, igg bajas (675mg/dl). tnius 5ng/l (negativas). cadenas kappa 33.6; cadenas lambda 29.1. ratio 1.155 (n). *serologias de atipicas: legionellla negativa, coxiella burneti igg/m negativas, chlamydia pneumoniae igg positiva con igm negativa, chlamydia psittaci igg e igm negativas. *pcr cmv: indetectable. *ca-125: 364u/ml. *inmunifijacion de cadenas libres en orina sin anormalidades evidentes. *analitica de control 02.12: leucocitos 6660/mm3, n 4800/mm3, l 1280/mm3, hb 8.8g/dl, hto 0.27, vcm 89fl, hcm 29pg, pqt 207000/mm3, inr 1.14 (tp: 79%), ttpar 1.01, cr 157umol/l, fge 26ml/min, urea 230mg/dl, na 132meq/l, k 4meq/l, a 31g/l, pcr 17mg/l. *analitica control 09.12 leucocitos 12810/mm3, n 11120/mm3, l 980/mm3, e 20/mm3, hb 8.9g/dl, hto 0.27,m vcm 88fl hcm 29pg, pqt 215000/mm3, cr 150umol/l, urea 43mg/dl, fge 28ml/min, na 128meq/l, k 3.7meq/l, ntprobnp 5207pg/ml * gsa basal 10.12: ph 7.44, pco2 33mmhg, po2 56mmhg, hco3 23mmol/l, o2 91%. *serologias agshb, anti-shb y anticore hb negativos. vih negativo. *serologias vhc positivas: rna-vhc indetectable. * tc de torax s/c 22.11: no se aprecian adenopatias supraclaviculares, axilares ni mediastinicas. cardiomegalia. ateromatosis coronario-aortica calcificada. arteria pulmonar de calibre normal, limitrofe. derrame pleural bilateral moderado y de baja densidad, con atelectasia pasiva del parenquima subyacente. no se aprecian consolidaciones parenquimatosas ni nodulos pulmonares de clara sospecha. opacidades seudonodulares subpleurales con tractos parenquimatosos asociados, parcialmente trigonales en ambos llii, de hasta 16 mm en segmento 6 izquierdo, inespecificas, de probable etiologia atelectasica (redonda), a controlar tras resolucion de cuadro agudo en al menos 6 meses. discretos infiltrados en vidrio deslustrado y finas opacidades intersticiales en llss y muy fino engrosamiento peribronquial, de probable etiologia infecciosa/inflamatoria, sin consolidaciones parenquimatosas establecidas. no se objetiva engrosamiento significativo de septos. no derrame pericardico. signos degenerativosgeneralizados con espondilosis dorsal, sin identificarse imagenes que sugieran lesiones secundarias. * ecografia abdominal 22.11: higado de tamaño en el limite superior de la normalidad, con ecoestructura homogenea y pequeños granulomas calcificados, sin otras lesiones focales evidentes. vena porta permeable con flujo hepatopetal. colecistectomia. ectasia leve de via biliar intra y extrahepatica, sin causa obstructiva evidente. pancreas parcialmente visualizado, sin alteraciones evidentes. bazo"
"""

input_text = "sumarize: No gross consolidation, atelectasis or infiltrate. No pleural fluid collection or pneumothorax. Cardiomediastinal silhouette is within normal limits. XXXX XXXX is intact."
input_text = "Lung volumes are XXXX. XXXX opacities are present in both lung bases. A hiatal hernia is present. Heart and pulmonary XXXX are normal."


model_id  = "/hhome/nlp2_g05/Asho_NLP/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
)
print("Model loaded")
print("Device:", generator.device)

tokens = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
print(len(tokens["input_ids"][0]))
generation = generator(
    input_text,
    eos_token_id=tokenizer.eos_token_id,
    min_length=5,
    max_new_tokens=int(len(tokens["input_ids"][0])*0.5)+6,
    num_beams=10,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True,
    no_repeat_ngram_size=2,
    use_cache=True,
    do_sample = True,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.95,
)

gen_text = generation[0]["generated_text"]
# Get the text after "The summary is:"
gen_text = gen_text.split("\nThe summary is:\n")[1]

# Get all the complete sentences
import nltk
gen_text_sentences = nltk.sent_tokenize(gen_text)


if len(gen_text_sentences) > 1:
    # Check if the last sentence is complete
    if gen_text_sentences[-1][-1] != ".":
        # Remove the last sentence
        gen_text_sentences = gen_text_sentences[:-1]

    # Check if the last sentence is only one number and a point
    if len(gen_text_sentences[-1]) == 2 and gen_text_sentences[-1][0].isdigit() and gen_text_sentences[-1][1] == ".":
        # Remove the last sentence
        gen_text_sentences = gen_text_sentences[:-1]

print(f"Result:\n{' '.join(gen_text_sentences)}")
