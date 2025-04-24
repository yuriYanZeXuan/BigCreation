from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json

def document_segmentation(input_str: str):
    """
    Document Segmentation
    """
    # Initialize the pipeline for document segmentation
    # Specify the task and model to be used
    p = pipeline(
        task=Tasks.document_segmentation,
        model='nlp_bert_document-segmentation_english-base')

    result = p(documents=input_str)
    texts = result[OutputKeys.TEXT]
    segments = texts.split('\n')  # Split by newline
    segments = [seg.strip() for seg in segments if seg.strip()]
    return segments

def read_txt_file(file_path: str) -> str:
    """
    Reads the content of a txt file and returns it as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def dataset_gen(segments):
    """
    Generates a dataset from the segmented text.
    """
    dataset = []
    instruction = "You will receive a text. Based on this text, generate a clear and concise question that is directly related to the most important and meaningful information in the content. The question should reflect the core message or key details of the text, and can be answered using information within the text. After generating the question, provide the corresponding answer based on the context. For example:\n\nInput:\nBeyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'.\n\nOutput:\nQuestion:When did Beyonce start becoming popular? Answer: In the late 1990s as lead singer of Destiny's Child."
    for segment in segments:
        # Here you can customize how to format each segment into a dataset entry
        dataset.append({
            'instruction': instruction,
            'input': segment,
            'output': ''  # Placeholder for output
        })
    with open('data/temp.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Specify the path to your .txt file
    file_path = 'test.txt'  # Replace with your actual file path
    input_str = read_txt_file(file_path)
    
    # Perform document segmentation
    segments = document_segmentation(input_str)
    
    dataset_gen(segments)
    
    # Print the segmented text
    # [print(f"Segment {i+1}: {seg}") for i, seg in enumerate(segments)]
