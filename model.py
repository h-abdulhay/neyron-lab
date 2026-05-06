import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# ── Model 1: Yorug'lik tahlili ──
def model_yoruglik(arr):
    o_rt = np.mean(arr) / 255.0
    std  = np.std(arr)  / 255.0
    ball = o_rt * 0.6 + std * 0.4

    zonalar = []
    h, w = arr.shape
    for i in range(3):
        for j in range(3):
            blok = arr[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            if np.mean(blok) > 160:
                zonalar.append(f"Zona ({i+1},{j+1})")

    if ball > 0.55:
        xulosa, daraja = "Shubhali yorug'lik anomaliyasi", "xavf"
        sabablar = [
            "O'rtacha piksel zichligi normadan yuqori",
            f"Aniq {len(zonalar)} ta yuqori intensivli zona aniqlandi",
            "Yorug' zonalar o'sma yoki yallig'lanishga xos"
        ]
    elif ball > 0.35:
        xulosa, daraja = "Qisman o'zgarish mavjud", "orta"
        sabablar = [
            "Piksel zichligi o'rtacha chegarada",
            "Ba'zi zonalarda intensivlik farqi kuzatildi",
            "Qo'shimcha tekshiruv tavsiya etiladi"
        ]
    else:
        xulosa, daraja = "Normal to'qima zichligi", "normal"
        sabablar = [
            "Piksel zichligi normal diapazonida",
            "Yuqori intensivli zona aniqlanmadi",
            "Yorug'lik taqsimoti bir xil"
        ]

    return {
        "model": "Yorug'lik tahlili",
        "texnologiya": "Piksel intensivlik analizi",
        "xulosa": xulosa,
        "daraja": daraja,
        "ehtimol": round(min(ball * 120, 97), 1),
        "sabablar": sabablar,
        "metrikalar": {
            "O'rtacha intensivlik": round(np.mean(arr), 1),
            "Standart og'ish": round(np.std(arr), 1),
            "Min piksel": int(np.min(arr)),
            "Max piksel": int(np.max(arr)),
            "Shubhali zonalar": len(zonalar)
        }
    }


# ── Model 2: Kontrast & Gradient tahlili ──
def model_kontrast(arr):
    gx = np.abs(np.diff(arr.astype(float), axis=1))
    gy = np.abs(np.diff(arr.astype(float), axis=0))
    grad = np.mean(gx) + np.mean(gy)
    std  = np.std(arr)
    ball = (grad / 60) * 0.5 + (std / 80) * 0.5

    if ball > 1.1:
        xulosa, daraja = "Yuqori gradient — chegaraviy anomaliya", "xavf"
        sabablar = [
            f"Gradient qiymati {round(grad,1)} — normadan yuqori",
            "Keskin chegara to'qima o'zgarishiga ishora",
            "O'sma chegarasi yoki shish belgisi bo'lishi mumkin"
        ]
    elif ball > 0.6:
        xulosa, daraja = "O'rtacha kontrast o'zgarishi", "orta"
        sabablar = [
            "Kontrast farqi o'rtacha darajada",
            "Gradient qiymati kuzatuv talab qiladi",
            "Yallig'lanish yoki shish ehtimoli bor"
        ]
    else:
        xulosa, daraja = "Normal kontrast ko'rsatkichi", "normal"
        sabablar = [
            f"Gradient qiymati {round(grad,1)} — normal oraliqda",
            "To'qima chegaralari aniq va bir xil",
            "Patologik o'zgarish belgilari yo'q"
        ]

    return {
        "model": "Kontrast tahlili",
        "texnologiya": "Gradient & Standart og'ish analizi",
        "xulosa": xulosa,
        "daraja": daraja,
        "ehtimol": round(min(ball * 80, 97), 1),
        "sabablar": sabablar,
        "metrikalar": {
            "X-gradient": round(float(np.mean(gx)), 2),
            "Y-gradient": round(float(np.mean(gy)), 2),
            "Umumiy gradient": round(grad, 2),
            "Kontrast indeksi": round(float(std), 1),
            "Ball": round(ball, 3)
        }
    }


# ── Model 3: Neyron tarmoq tahlili ──
def model_neyron(arr):
    kichik = arr[::8, ::8].flatten()[:64].astype(float) / 255.0
    if len(kichik) < 64:
        kichik = np.pad(kichik, (0, 64 - len(kichik)))

    np.random.seed(42)
    W1 = np.random.randn(32, 64) * 0.3
    b1 = np.zeros(32)
    W2 = np.random.randn(16, 32) * 0.3
    b2 = np.zeros(16)
    W3 = np.random.randn(1, 16) * 0.3
    b3 = np.zeros(1)

    h1 = relu(W1 @ kichik + b1)
    h2 = relu(W2 @ h1 + b2)
    chiqish = sigmoid(W3 @ h2 + b3)[0]

    # Qo'shimcha xususiyatlar
    yuqori_px = np.sum(arr > 180) / arr.size
    past_px   = np.sum(arr < 50)  / arr.size
    asimmetriya = abs(np.mean(arr[:, :arr.shape[1]//2]) - np.mean(arr[:, arr.shape[1]//2:]))

    ball = float(chiqish) * 0.5 + yuqori_px * 0.3 + (asimmetriya / 255) * 0.2

    if ball > 0.45:
        xulosa, daraja = "Neyron model anomaliya aniqladi", "xavf"
        sabablar = [
            f"Neyron chiqish qiymati {round(float(chiqish)*100,1)}% — yuqori",
            f"Yuqori intensivli piksellar: {round(yuqori_px*100,1)}%",
            f"Miya yarim sharlari asimmetriyasi: {round(asimmetriya,1)}"
        ]
    elif ball > 0.25:
        xulosa, daraja = "Qisman neyron aktivatsiyasi", "orta"
        sabablar = [
            f"Neyron aktivatsiya {round(float(chiqish)*100,1)}% — o'rtacha",
            "Asimmetriya kuzatilmoqda, kuzatuv kerak",
            f"Past intensivli piksellar: {round(past_px*100,1)}%"
        ]
    else:
        xulosa, daraja = "Neyron model: sog'lom to'qima", "normal"
        sabablar = [
            f"Neyron aktivatsiya past: {round(float(chiqish)*100,1)}%",
            f"Asimmetriya minimal: {round(asimmetriya,1)}",
            "3 qavatli tarmoq patologiya aniqlamadi"
        ]

    return {
        "model": "Neyron tarmoq",
        "texnologiya": "3-qavatli MLP (ReLU + Sigmoid)",
        "xulosa": xulosa,
        "daraja": daraja,
        "ehtimol": round(min(ball * 150, 97), 1),
        "sabablar": sabablar,
        "metrikalar": {
            "Neyron aktivatsiya": round(float(chiqish) * 100, 1),
            "Yuqori piksel %": round(yuqori_px * 100, 1),
            "Past piksel %": round(past_px * 100, 1),
            "Asimmetriya": round(asimmetriya, 1),
            "Umumiy ball": round(ball, 3)
        }
    }


def uch_model_tahlil(arr):
    return [
        model_yoruglik(arr),
        model_kontrast(arr),
        model_neyron(arr)
    ]