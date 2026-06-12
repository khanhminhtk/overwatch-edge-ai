from pathlib import Path
import warnings
from google.cloud import vision

warnings.filterwarnings("ignore", category=FutureWarning)

SERVICE_ACCOUNT_PATH = (
    "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/config/"
    "sincere-sun-493304-i7-8402163d2d44.json"
)

IMAGE_PATHS = [
    "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/tmp/470163109_122094372572659902_3687873334305359309_n.jpg",
    "/home/minhtk/code/overwatch-edge-ai/worktree/backend_nexus/services/model-lifecycle-service/tmp/img1849-17484214382622019899781-0-0-831-1330-crop-1748421608176207533437.webp",
]

client = vision.ImageAnnotatorClient.from_service_account_json(
    SERVICE_ACCOUNT_PATH
)


def extract_words_from_response(response, image_path: str):
    if response.error.message:
        return {
            "image_path": image_path,
            "success": False,
            "error": response.error.message,
            "words": [],
        }

    words = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(symbol.text for symbol in word.symbols)

                    bbox = [
                        {"x": vertex.x, "y": vertex.y}
                        for vertex in word.bounding_box.vertices
                    ]

                    words.append(
                        {
                            "text": text,
                            "bbox": bbox,
                            "confidence": word.confidence,
                        }
                    )

    return {
        "image_path": image_path,
        "success": True,
        "error": None,
        "words": words,
    }


def build_request(image_path: str):
    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)

    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
    )

    image_context = vision.ImageContext(
        language_hints=["vi"]
    )

    return vision.AnnotateImageRequest(
        image=image,
        features=[feature],
        image_context=image_context,
    )


requests = [build_request(path) for path in IMAGE_PATHS]

batch_response = client.batch_annotate_images(
    requests=requests
)

results = []

for image_path, response in zip(IMAGE_PATHS, batch_response.responses):
    result = extract_words_from_response(response, image_path)
    results.append(result)

for result in results:
    print("IMAGE:", result["image_path"])
    print("SUCCESS:", result["success"])

    if not result["success"]:
        print("ERROR:", result["error"])
        continue

    for word in result["words"]:
        print(word)

    print("-" * 80)