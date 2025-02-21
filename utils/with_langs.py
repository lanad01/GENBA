from IPython.display import Image, display

MODEL = 'gpt-4o-mini'
MODEL_FINETUNED = 'ft:gpt-4o-mini-2024-07-18:quintet::AVCuZwL4'
MODEL_HAIKU = 'claude-3-haiku'
EMBEDDING_MODEL = 'text-embedding-3-small'

def draw_graph(graph):
    """compile된 그래프 이미지 출력"""
    display(Image(graph.get_graph().draw_mermaid_png()))

def stream_structured_output(stream, attr):
    """stream 객체 리턴 속성이 하나 뿐일 때만 사용 가능함."""
    last_chunk = ""
    for chunk in stream:
        cur_chunk = getattr(chunk, attr)
        diff = cur_chunk[len(last_chunk):]
        print(diff, end="", flush=True)
        last_chunk = cur_chunk
    return last_chunk

if __name__ == "__main__":
    pass