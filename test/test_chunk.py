import sys
sys.path.insert(0, "/home/bahung/DoAn")

from core.rag.chunk import chunk_markdown_file

test_file = "data/data_process/chuong_trinh_dao_tao/1.1. Kỹ thuật Cơ điện tử.md"

print("=" * 70)
print(f" File: {test_file}")
print("=" * 70)

# Now returns List[BaseNode] instead of List[Dict]
nodes = chunk_markdown_file(test_file)

print(f"\n Total nodes: {len(nodes)}\n")

for i, node in enumerate(nodes):
    content = node.get_content()
    metadata = node.metadata
    
    print(f"\n{'─' * 70}")
    print(f" NODE #{i}")
    print(f"   Type: {type(node).__name__}")
    print(f"   Length: {len(content)} chars")
    if metadata:
        print(f"   Metadata: {metadata}")
    print(f"{'─' * 70}")
    content_preview = content[:200]
    if len(content) > 200:
        content_preview += "..."
    print(content_preview)

with open("test_chunk.md", "w", encoding="utf-8") as f:
    for i, node in enumerate(nodes):
        content = node.get_content()
        metadata = node.metadata
        
        f.write(f"# NODE {i}\n")
        f.write(f"**Type:** {type(node).__name__}\n\n")
        f.write("**Metadata:**\n")
        for key, value in metadata.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n**Content:**\n")
        f.write(content)
        f.write("\n\n---\n\n")

print("\n Done")
