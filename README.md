![zxvdzxxfvgdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/BpfY3BKv4KtfZL2gRLquF.png)

# **Clipart-126-DomainNet**

> **Clipart-126-DomainNet** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify clipart images into 126 domain categories using the **SiglipForImageClassification** architecture.

![- visual selection(3).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MvDgSuvhK2mGBgUl8gcnO.png)

*Moment Matching for Multi-Source Domain Adaptation* : https://arxiv.org/pdf/1812.01754

*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
                         precision    recall  f1-score   support

       aircraft_carrier     0.8667    0.4643    0.6047        56
            alarm_clock     0.9706    0.8919    0.9296        74
                    ant     0.8889    0.8615    0.8750        65
                  anvil     0.5984    0.6083    0.6033       120
              asparagus     0.8158    0.6078    0.6966        51
                    axe     0.7544    0.5309    0.6232        81
                 banana     0.7111    0.5517    0.6214        58
                 basket     0.8571    0.8182    0.8372        66
                bathtub     0.7531    0.7821    0.7673        78
                   bear     0.9118    0.6458    0.7561        48
                    bee     0.9636    0.9636    0.9636       165
                   bird     0.8967    0.9529    0.9240       255
             blackberry     0.8082    0.8429    0.8252        70
              blueberry     0.8661    0.8981    0.8818       108
              bottlecap     0.7821    0.8299    0.8053       147
               broccoli     0.8947    0.8947    0.8947        95
                    bus     0.9663    0.9348    0.9503        92
              butterfly     0.9333    0.9545    0.9438       132
                 cactus     0.9677    0.9091    0.9375        99
                   cake     0.8750    0.8099    0.8412       121
             calculator     0.9583    0.5897    0.7302        39
                  camel     0.9391    0.9310    0.9351       116
                 camera     0.8846    0.8679    0.8762        53
                 candle     0.8298    0.8478    0.8387        92
                 cannon     0.8551    0.8551    0.8551        69
                  canoe     0.8462    0.7432    0.7914        74
                 carrot     0.8800    0.7719    0.8224        57
                 castle     1.0000    0.8511    0.9195        47
                    cat     0.8167    0.7903    0.8033        62
            ceiling_fan     1.0000    0.2000    0.3333        30
             cell_phone     0.7400    0.6491    0.6916        57
                  cello     0.8372    0.9114    0.8727        79
                  chair     0.8986    0.8378    0.8671        74
             chandelier     0.9617    0.9263    0.9437       190
             coffee_cup     0.8811    0.9389    0.9091       229
                compass     0.9799    0.9012    0.9389       162
               computer     0.7124    0.9045    0.7970       178
                    cow     0.9517    0.9718    0.9617       142
                   crab     0.8738    0.9000    0.8867       100
              crocodile     0.9778    0.9167    0.9462       144
            cruise_ship     0.8544    0.9072    0.8800       194
                    dog     0.8125    0.7761    0.7939        67
                dolphin     0.7680    0.7500    0.7589       128
                 dragon     0.9512    0.9176    0.9341        85
                  drums     0.8919    0.9635    0.9263       137
                   duck     0.8774    0.8447    0.8608       161
               dumbbell     0.9048    0.9500    0.9268       280
               elephant     0.9038    0.8952    0.8995       105
             eyeglasses     0.8636    0.8488    0.8562       291
                feather     0.8564    0.9227    0.8883       181
                  fence     0.9211    0.8400    0.8787       125
                   fish     0.8963    0.8768    0.8864       138
               flamingo     0.9636    0.9381    0.9507       226
                 flower     0.9146    0.9454    0.9298       238
                   foot     0.8780    0.8889    0.8834        81
                   fork     0.9032    0.9091    0.9061       154
                   frog     0.9420    0.9489    0.9455       137
                giraffe     0.9643    0.9153    0.9391       118
                 goatee     0.8763    0.9422    0.9081       173
                 grapes     0.9114    0.8571    0.8834        84
                 guitar     0.9595    0.8554    0.9045        83
                 hammer     0.6111    0.7719    0.6822       114
             helicopter     0.9444    0.9533    0.9488       107
                 helmet     0.7368    0.8550    0.7915       131
                  horse     0.9588    0.9819    0.9702       166
               kangaroo     0.9125    0.8488    0.8795        86
                lantern     0.8254    0.7536    0.7879        69
                 laptop     0.8108    0.5000    0.6186        60
                   leaf     0.7143    0.3333    0.4545        30
                   lion     0.9744    0.8085    0.8837        47
               lipstick     0.7875    0.6632    0.7200        95
                lobster     0.8963    0.9130    0.9046       161
             microphone     0.7925    0.9231    0.8528        91
                 monkey     0.9623    0.9027    0.9315       113
               mosquito     0.8636    0.8444    0.8539        45
                  mouse     0.9167    0.8333    0.8730        66
                    mug     0.8989    0.8163    0.8556        98
               mushroom     0.9429    0.9429    0.9429       105
                  onion     0.9365    0.8429    0.8872       140
                  panda     1.0000    0.9726    0.9861        73
                 peanut     0.5900    0.7195    0.6484        82
                   pear     0.7692    0.7246    0.7463        69
                   peas     0.8000    0.7429    0.7704        70
                 pencil     0.6667    0.0909    0.1600        44
                penguin     0.9717    0.9279    0.9493       111
                    pig     0.9551    0.8252    0.8854       103
                 pillow     0.6290    0.5571    0.5909        70
              pineapple     0.9846    0.8889    0.9343        72
                 potato     0.6038    0.6531    0.6275        98
           power_outlet     0.8636    0.4043    0.5507        47
                  purse     0.0000    0.0000    0.0000        27
                 rabbit     0.9341    0.8586    0.8947        99
                raccoon     0.8836    0.9021    0.8927       143
             rhinoceros     0.8750    0.9459    0.9091        74
                  rifle     0.7595    0.7500    0.7547        80
              saxophone     0.9454    0.9886    0.9665       175
            screwdriver     0.7521    0.6929    0.7213       127
             sea_turtle     0.9677    0.9626    0.9651       187
                see_saw     0.6679    0.8698    0.7556       215
                  sheep     0.9355    0.9158    0.9255        95
                   shoe     0.8969    0.8700    0.8832       100
             skateboard     0.8632    0.8673    0.8652       211
                  snake     0.9302    0.9160    0.9231       131
              speedboat     0.8187    0.8976    0.8563       166
                 spider     0.9043    0.9286    0.9163       112
               squirrel     0.7945    0.8855    0.8375       131
             strawberry     0.8687    0.9923    0.9264       260
            streetlight     0.8178    0.9293    0.8700       198
            string_bean     0.8525    0.8000    0.8254        65
              submarine     0.8022    0.8902    0.8439       164
                   swan     0.8397    0.9003    0.8690       291
                  table     0.8564    0.9200    0.8871       175
                 teapot     0.8763    0.9189    0.8971       185
             teddy-bear     0.9006    0.8953    0.8980       172
             television     0.8509    0.8220    0.8362       118
       the_Eiffel_Tower     0.9468    0.9082    0.9271        98
the_Great_Wall_of_China     0.9462    0.9462    0.9462        93
                  tiger     0.9417    0.9826    0.9617       230
                    toe     0.8250    0.6600    0.7333        50
                  train     0.9362    0.9778    0.9565        90
                  truck     0.9367    0.8916    0.9136        83
               umbrella     0.9633    0.9545    0.9589       110
                   vase     0.7642    0.8393    0.8000       112
             watermelon     0.9527    0.9527    0.9527       148
                  whale     0.7453    0.8144    0.7783       194
                  zebra     0.9275    0.9676    0.9471       185

               accuracy                         0.8691     14818
              macro avg     0.8613    0.8251    0.8351     14818
           weighted avg     0.8705    0.8691    0.8661     14818
```



The model categorizes images into the following 126 classes:
- **Class 0:** "aircraft_carrier"  
- **Class 1:** "alarm_clock"  
- **Class 2:** "ant"  
- **Class 3:** "anvil"  
- **Class 4:** "asparagus"  
- **Class 5:** "axe"  
- **Class 6:** "banana"  
- **Class 7:** "basket"  
- **Class 8:** "bathtub"  
- **Class 9:** "bear"  
- **Class 10:** "bee"  
- **Class 11:** "bird"  
- **Class 12:** "blackberry"  
- **Class 13:** "blueberry"  
- **Class 14:** "bottlecap"  
- **Class 15:** "broccoli"  
- **Class 16:** "bus"  
- **Class 17:** "butterfly"  
- **Class 18:** "cactus"  
- **Class 19:** "cake"  
- **Class 20:** "calculator"  
- **Class 21:** "camel"  
- **Class 22:** "camera"  
- **Class 23:** "candle"  
- **Class 24:** "cannon"  
- **Class 25:** "canoe"  
- **Class 26:** "carrot"  
- **Class 27:** "castle"  
- **Class 28:** "cat"  
- **Class 29:** "ceiling_fan"  
- **Class 30:** "cell_phone"  
- **Class 31:** "cello"  
- **Class 32:** "chair"  
- **Class 33:** "chandelier"  
- **Class 34:** "coffee_cup"  
- **Class 35:** "compass"  
- **Class 36:** "computer"  
- **Class 37:** "cow"  
- **Class 38:** "crab"  
- **Class 39:** "crocodile"  
- **Class 40:** "cruise_ship"  
- **Class 41:** "dog"  
- **Class 42:** "dolphin"  
- **Class 43:** "dragon"  
- **Class 44:** "drums"  
- **Class 45:** "duck"  
- **Class 46:** "dumbbell"  
- **Class 47:** "elephant"  
- **Class 48:** "eyeglasses"  
- **Class 49:** "feather"  
- **Class 50:** "fence"  
- **Class 51:** "fish"  
- **Class 52:** "flamingo"  
- **Class 53:** "flower"  
- **Class 54:** "foot"  
- **Class 55:** "fork"  
- **Class 56:** "frog"  
- **Class 57:** "giraffe"  
- **Class 58:** "goatee"  
- **Class 59:** "grapes"  
- **Class 60:** "guitar"  
- **Class 61:** "hammer"  
- **Class 62:** "helicopter"  
- **Class 63:** "helmet"  
- **Class 64:** "horse"  
- **Class 65:** "kangaroo"  
- **Class 66:** "lantern"  
- **Class 67:** "laptop"  
- **Class 68:** "leaf"  
- **Class 69:** "lion"  
- **Class 70:** "lipstick"  
- **Class 71:** "lobster"  
- **Class 72:** "microphone"  
- **Class 73:** "monkey"  
- **Class 74:** "mosquito"  
- **Class 75:** "mouse"  
- **Class 76:** "mug"  
- **Class 77:** "mushroom"  
- **Class 78:** "onion"  
- **Class 79:** "panda"  
- **Class 80:** "peanut"  
- **Class 81:** "pear"  
- **Class 82:** "peas"  
- **Class 83:** "pencil"  
- **Class 84:** "penguin"  
- **Class 85:** "pig"  
- **Class 86:** "pillow"  
- **Class 87:** "pineapple"  
- **Class 88:** "potato"  
- **Class 89:** "power_outlet"  
- **Class 90:** "purse"  
- **Class 91:** "rabbit"  
- **Class 92:** "raccoon"  
- **Class 93:** "rhinoceros"  
- **Class 94:** "rifle"  
- **Class 95:** "saxophone"  
- **Class 96:** "screwdriver"  
- **Class 97:** "sea_turtle"  
- **Class 98:** "see_saw"  
- **Class 99:** "sheep"  
- **Class 100:** "shoe"  
- **Class 101:** "skateboard"  
- **Class 102:** "snake"  
- **Class 103:** "speedboat"  
- **Class 104:** "spider"  
- **Class 105:** "squirrel"  
- **Class 106:** "strawberry"  
- **Class 107:** "streetlight"  
- **Class 108:** "string_bean"  
- **Class 109:** "submarine"  
- **Class 110:** "swan"  
- **Class 111:** "table"  
- **Class 112:** "teapot"  
- **Class 113:** "teddy-bear"  
- **Class 114:** "television"  
- **Class 115:** "the_Eiffel_Tower"  
- **Class 116:** "the_Great_Wall_of_China"  
- **Class 117:** "tiger"  
- **Class 118:** "toe"  
- **Class 119:** "train"  
- **Class 120:** "truck"  
- **Class 121:** "umbrella"  
- **Class 122:** "vase"  
- **Class 123:** "watermelon"  
- **Class 124:** "whale"  
- **Class 125:** "zebra"

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```
```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Clipart-126-DomainNet"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def clipart_classification(image):
    """Predicts the clipart category for an input image."""
    # Convert the input numpy array to a PIL Image and ensure it's in RGB format
    image = Image.fromarray(image).convert("RGB")
    
    # Process the image and prepare it for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply softmax to obtain probabilities for each class
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Mapping from indices to clipart category labels
    labels = {
        "0": "aircraft_carrier", "1": "alarm_clock", "2": "ant", "3": "anvil", "4": "asparagus",
        "5": "axe", "6": "banana", "7": "basket", "8": "bathtub", "9": "bear",
        "10": "bee", "11": "bird", "12": "blackberry", "13": "blueberry", "14": "bottlecap",
        "15": "broccoli", "16": "bus", "17": "butterfly", "18": "cactus", "19": "cake",
        "20": "calculator", "21": "camel", "22": "camera", "23": "candle", "24": "cannon",
        "25": "canoe", "26": "carrot", "27": "castle", "28": "cat", "29": "ceiling_fan",
        "30": "cell_phone", "31": "cello", "32": "chair", "33": "chandelier", "34": "coffee_cup",
        "35": "compass", "36": "computer", "37": "cow", "38": "crab", "39": "crocodile",
        "40": "cruise_ship", "41": "dog", "42": "dolphin", "43": "dragon", "44": "drums",
        "45": "duck", "46": "dumbbell", "47": "elephant", "48": "eyeglasses", "49": "feather",
        "50": "fence", "51": "fish", "52": "flamingo", "53": "flower", "54": "foot",
        "55": "fork", "56": "frog", "57": "giraffe", "58": "goatee", "59": "grapes",
        "60": "guitar", "61": "hammer", "62": "helicopter", "63": "helmet", "64": "horse",
        "65": "kangaroo", "66": "lantern", "67": "laptop", "68": "leaf", "69": "lion",
        "70": "lipstick", "71": "lobster", "72": "microphone", "73": "monkey", "74": "mosquito",
        "75": "mouse", "76": "mug", "77": "mushroom", "78": "onion", "79": "panda",
        "80": "peanut", "81": "pear", "82": "peas", "83": "pencil", "84": "penguin",
        "85": "pig", "86": "pillow", "87": "pineapple", "88": "potato", "89": "power_outlet",
        "90": "purse", "91": "rabbit", "92": "raccoon", "93": "rhinoceros", "94": "rifle",
        "95": "saxophone", "96": "screwdriver", "97": "sea_turtle", "98": "see_saw", "99": "sheep",
        "100": "shoe", "101": "skateboard", "102": "snake", "103": "speedboat", "104": "spider",
        "105": "squirrel", "106": "strawberry", "107": "streetlight", "108": "string_bean",
        "109": "submarine", "110": "swan", "111": "table", "112": "teapot", "113": "teddy-bear",
        "114": "television", "115": "the_Eiffel_Tower", "116": "the_Great_Wall_of_China",
        "117": "tiger", "118": "toe", "119": "train", "120": "truck", "121": "umbrella",
        "122": "vase", "123": "watermelon", "124": "whale", "125": "zebra"
    }
    
    # Create a dictionary mapping each label to its corresponding probability (rounded)
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=clipart_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Clipart-126-DomainNet Classification",
    description="Upload a clipart image to classify it into one of 126 domain categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```
---

# **Intended Use:**

The **Clipart-126-DomainNet** model is designed for clipart image classification. It categorizes clipart images into a wide range of domains—from objects like an "aircraft_carrier" or "alarm_clock" to various everyday items. Potential use cases include:

- **Digital Art and Design:** Assisting designers in organizing and retrieving clipart assets.
- **Content Management:** Enhancing digital asset management systems with robust clipart classification.
- **Creative Search Engines:** Enabling clipart-based search for design inspiration and resource curation.
- **Computer Vision Research:** Serving as a benchmark for studies in clipart recognition and domain adaptation.
