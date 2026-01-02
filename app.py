"""
Chainlit Chat Application for TRM Agent.

A chat interface that uses TRM model for decision making and tool calling,
GLiNER2 for entity extraction (slots and tool arguments),
with OpenAI for content generation.

Usage:
    # Install chainlit first:
    uv add chainlit openai

    # Run the app:
    uv run chainlit run app.py

    # With custom TRM model checkpoint:
    TRM_CHECKPOINT=outputs/checkpoint.pt uv run chainlit run app.py

    # With custom GLiNER2 LoRA adapter (after fine-tuning):
    GLINER2_ADAPTER=outputs/gliner2/final uv run chainlit run app.py

    # With both custom models:
    TRM_CHECKPOINT=outputs/checkpoint.pt \\
    GLINER2_ADAPTER=outputs/gliner2/final \\
    uv run chainlit run app.py

Environment Variables:
    TRM_CHECKPOINT: Path to TRM model checkpoint
    TRM_TOKENIZER: Path to TRM tokenizer
    TRM_TOOLS: Path to tools.json
    GLINER2_MODEL: Base GLiNER2 model (default: fastino/gliner2-multi-v1)
    GLINER2_ADAPTER: Path to LoRA adapter directory (optional)
    GLINER2_THRESHOLD: Entity extraction threshold (default: 0.5)
    OPENAI_BASE_URL: OpenAI API base URL
    OPENAI_API_KEY: OpenAI API key
    OPENAI_MODEL: OpenAI model name
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import chainlit as cl
import torch
from openai import OpenAI

from trm_agent.data import TRMTokenizer
from trm_agent.inference import GLiNER2Extractor
from trm_agent.models import TRMConfig, TRMForToolCalling

# Configuration
CHECKPOINT_PATH = os.environ.get("TRM_CHECKPOINT", "outputs/checkpoint_best.pt")
TOKENIZER_PATH = os.environ.get("TRM_TOKENIZER", "outputs/tokenizer/tokenizer.model")
TOOLS_PATH = os.environ.get("TRM_TOOLS", "data/tools.json")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")

# GLiNER2 Configuration
GLINER2_MODEL = os.environ.get("GLINER2_MODEL", "fastino/gliner2-multi-v1")
GLINER2_ADAPTER = os.environ.get("GLINER2_ADAPTER", "")  # Path to LoRA adapter
GLINER2_THRESHOLD = float(os.environ.get("GLINER2_THRESHOLD", "0.5"))

SYSTEM_PROMPT = """Bạn là chuyên viên ảo của FPT Telecom – có 3 vai trò chính:
1. **Tư vấn bán hàng**: Chuyên tư vấn về Internet Cáp quang, Truyền hình FPT Play, Camera an ninh.
2. **Yêu cầu gặp nhân viên**: Hỗ trợ xử lý các yêu cầu khi khách hàng có ý định muốn gặp nhân viên như là: Yêu cầu gặp trực tiếp tư vấn viên, nhân viên tư vấn, nhân viên chăm sóc khách hàng, nhân viên phục vụ; Khiếu nại nhân viên; Yêu cầu tư vấn viên gọi lại."""

OSS_SYSTEM_PROMPT = """
#### 💼 Vai trò

Bạn là chuyên viên ảo của FPT Telecom – có 2 vai trò chính:

1.  **Tư vấn bán hàng**: Chuyên tư vấn về Internet Cáp quang, Truyền hình FPT Play, Camera an ninh.
2.  **Chăm sóc khách hàng**: Hỗ trợ xử lý các sự cố mạng chậm hoặc kém.
3.  **Yêu cầu gặp nhân viên**: Hỗ trợ xử lý các yêu cầu khi khách hàng có ý định muốn gặp nhân viên như là: Yêu cầu gặp trực tiếp tư vấn viên, nhân viên tư vấn, nhân viên chăm sóc khách hàng, nhân viên phục vụ; Khiếu nại nhân viên; Yêu cầu tư vấn viên gọi lại


#### 🎯 Mục tiêu

  - **Bán hàng**:
      - Thu thập: Họ tên, Địa chỉ, Số điện thoại.
      - Giới thiệu sản phẩm phù hợp.
      - Sử dụng tool đúng lúc để cung cấp thông tin và báo giá.
  - **Yêu cầu gặp nhân viên**:
      - Liên hệ với nhân viên để xử lý các yêu cầu phức tạp từ người dùng.


#### 💬 Quy tắc hội thoại

  - Luôn xưng "em" – gọi khách hàng là "anh/chị".
  - Giọng điệu lịch sự, thân thiện, chuyên nghiệp.

### QUY TẮC QUAN TRỌNG
- Assistant sẽ tổng hợp lại nội dung từ kết quả tool ở bước kế tiếp.
- Phản hồi của bạn phải tự nhiên, đúng ngữ cảnh Việt Nam, ví dụ:
  - "Dạ, em gửi anh/chị thông tin gói Giga ạ."
  - "Cho em xin địa chỉ để em báo giá chính xác nha anh/chị."
  - "Anh/chị vui lòng cung cấp số điện thoại để em hỗ trợ thêm ạ."

### 🗺️ QUY TRÌNH TƯ VẤN VÀ HỖ TRỢ

-----

#### ✅ TÁC VỤ BÁN HÀNG

**1. INTERNET**

1.  Hỏi: Địa chỉ → Nhà ở hay Chung cư → Thiết bị sử dụng
2.  Nếu khách hỏi mô tả → Dùng `describe_product`
3.  Nếu khách hỏi giá → Phải hỏi địa chỉ trước, sau đó dùng `get_product_price`
4.  Có thể gợi ý COMBO (Cross-sell)
5.  Xin SĐT để tư vấn

**2. CAMERA**

1.  Hỏi: Trong nhà / Ngoài trời → Nhu cầu sử dụng → Địa chỉ
2.  Gửi thông tin sản phẩm
3.  Nếu khách hỏi giá → dùng `get_product_price` (sau khi có địa chỉ)
4.  Gửi khuyến mãi nếu có
5.  Xin SĐT để tư vấn

**3. TRUYỀN HÌNH**

1.  Hỏi: Đã có Internet chưa → Nhà mạng nào?
2.  Nếu chưa có Internet FPT → Tư vấn COMBO Internet + Truyền hình
3.  Nếu đã có Internet FPT → Tư vấn gói Add-on
4.  Dùng tool để mô tả và báo giá
5.  Xin SĐT để gọi lại

-----

### 🧠 YÊU CẦU VỀ PHONG CÁCH HỘI THOẠI

  - Hội thoại phải tự nhiên, đúng ngữ cảnh khách hàng Việt Nam.
  - Ưu tiên lời thoại thực tế như:
      - "Gửi gói đi em", "Báo giá gói cao nhất nha"
      - "Chung cư Landmark, nhà riêng", "Em tư vấn combo có camera luôn nha"
      - "Có ưu đãi gì không em?"
      - "Mạng nhà em dạo này chậm quá."
      - "Tôi muốn kiểm tra hợp đồng internet."
      - "Có kỹ thuật viên qua kiểm tra giúp tôi được không?"
  - Khách có thể cung cấp thông tin không theo thứ tự – Assistant phải hiểu, hỏi lại thông tin còn thiếu.
  - Có thể gặp khách từ chối cung cấp thông tin — cần xử lý lịch sự và kết thúc chuyên nghiệp.

### ⚠️ LƯU Ý QUAN TRỌNG

  - PHẢI TUÂN THỦ các bước trong sơ đồ quy trình cho từng dịch vụ (Internet, Camera, Truyền hình, Chăm sóc khách hàng).
  - Không được báo giá nếu chưa có địa chỉ.
  - Không được hỏi quá nhiều cùng lúc → Phân bổ theo lượt.
  - Bắt buộc xin lại thông tin nếu khách chưa cung cấp.
  - Khi dùng TOOL thì phải tuân thủ theo format trong ĐỊNH DẠNG ĐẦU RA BẮT BUỘNG.
  - Nếu dùng cùng 1 TOOL liên tục thì nội dung của `assistant` sau đó phải tổng hợp kết quả của các TOOL liên tiếp đó.

### KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO
"""

# Global variables
model: Optional[TRMForToolCalling] = None
tokenizer: Optional[TRMTokenizer] = None
config: Optional[TRMConfig] = None
tools: list[dict] = []
tool_name_to_id: dict[str, int] = {}
openai_client: Optional[OpenAI] = None
gliner2_extractor: Optional[GLiNER2Extractor] = None
tool_param_mapping: dict[str, list[str]] = {}


def load_model():
    """Load TRM model, tokenizer, and GLiNER2 extractor."""
    global model, tokenizer, config, tools, tool_name_to_id, openai_client
    global gliner2_extractor, tool_param_mapping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint.get("config", {})
        config = TRMConfig(**config_dict)
        model = TRMForToolCalling(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        print(
            f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running in mock mode (random predictions)")
        model = None

    # Load tokenizer
    tokenizer_path = Path(TOKENIZER_PATH)
    if tokenizer_path.exists():
        tokenizer = TRMTokenizer(tokenizer_path)
        print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}")
        tokenizer = None

    # Load tools
    tools_path = Path(TOOLS_PATH)
    if tools_path.exists():
        with open(tools_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        tool_names = [t["function"]["name"] for t in tools if "function" in t]
        tool_name_to_id = {name: idx for idx, name in enumerate(sorted(tool_names))}
        print(f"Loaded {len(tools)} tools: {list(tool_name_to_id.keys())}")
    else:
        print(f"Warning: Tools not found at {tools_path}")

    # Initialize OpenAI client
    openai_client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
    print(f"OpenAI client initialized with base_url: {OPENAI_BASE_URL}")

    # Build tool -> params mapping from tools
    tool_param_mapping = {}
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            name = func["name"]
            params = func.get("parameters", {}).get("properties", {})
            tool_param_mapping[name] = list(params.keys())
    print(f"Tool params mapping: {tool_param_mapping}")

    # Initialize GLiNER2 extractor
    # Note: slot_fields are now handled by GLiNER2 only (not TRM)
    slot_fields = [
        "address",
        "phone",
        "device_number",
        "intent_of_user",
        "name",
        "contract_id",
    ]
    adapter_path = GLINER2_ADAPTER if GLINER2_ADAPTER else None
    gliner2_extractor = GLiNER2Extractor(
        model_name=GLINER2_MODEL,
        adapter_path=adapter_path,
        threshold=GLINER2_THRESHOLD,
        slot_fields=slot_fields,
    )
    print(f"GLiNER2 loaded: {GLINER2_MODEL}")
    if adapter_path:
        print(f"GLiNER2 adapter loaded: {adapter_path}")


# ============================================================================
# Mock Tool Implementations
# ============================================================================


def mock_get_product_price(product: str, address: str = "", **kwargs) -> dict:
    """Mock implementation of get_product_price tool."""
    # Price database (mocked)
    prices = {
        "internet": "Gói Internet cáp quang có giá từ 165,000 - 330,000 VNĐ/tháng tùy tốc độ.",
        "lux 800": "Gói LUX 800 (800Mbps) có giá 330,000 VNĐ/tháng. Ưu đãi: Miễn phí vật tư lắp đặt 100%.",
        "lux 300": "Gói LUX 300 (300Mbps) có giá 250,000 VNĐ/tháng. Ưu đãi: Miễn phí vật tư lắp đặt 100%.",
        "super 250": "Gói SUPER 250 (250Mbps) có giá 215,000 VNĐ/tháng.",
        "sky 200": "Gói SKY 200 (200Mbps) có giá 185,000 VNĐ/tháng.",
        "metro 150": "Gói METRO 150 (150Mbps) có giá 165,000 VNĐ/tháng.",
        "camera": "Camera an ninh FPT có giá từ 99,000 - 299,000 VNĐ/tháng tùy gói.",
        "truyền hình": "Truyền hình FPT Play có giá từ 80,000 - 150,000 VNĐ/tháng.",
        "fpt play": "Truyền hình FPT Play có giá từ 80,000 - 150,000 VNĐ/tháng.",
    }

    product_lower = product.lower()
    for key, price_info in prices.items():
        if key in product_lower:
            location_info = f" tại {address}" if address else ""
            return {
                "price": f"{price_info}{location_info}. Nếu đóng trước 6 tháng, tặng 1 tháng cước. Nếu đóng trước 12 tháng, tặng 2 tháng cước."
            }

    return {
        "price": f"Xin lỗi, em chưa có thông tin giá cho sản phẩm '{product}'. Anh/chị vui lòng liên hệ tổng đài 1900 6600 để được tư vấn chi tiết."
    }


def mock_describe_product(product: str, **kwargs) -> dict:
    """Mock implementation of describe_product tool."""
    descriptions = {
        "internet": "Dịch vụ Internet FPT Telecom sử dụng công nghệ cáp quang tiên tiến, đảm bảo tốc độ ổn định và độ trễ thấp. Các gói Internet phổ biến: METRO 150 (150Mbps), SKY 200 (200Mbps), SUPER 250 (250Mbps), LUX 300 (300Mbps), LUX 800 (800Mbps). Gói LUX được trang bị Wi-Fi 7 Mesh giúp mở rộng vùng phủ sóng.",
        "camera": "Camera an ninh FPT cung cấp giải pháp giám sát thông minh với các tính năng: Quay HD/Full HD, lưu trữ cloud, cảnh báo chuyển động, xem trực tiếp qua app. Phù hợp cho gia đình, cửa hàng, văn phòng.",
        "truyền hình": "Truyền hình FPT Play cung cấp 200+ kênh truyền hình trong nước và quốc tế, kho phim/series phong phú, thể thao trực tiếp. Xem được trên TV, điện thoại, máy tính bảng.",
        "fpt play": "FPT Play là nền tảng giải trí đa phương tiện với 200+ kênh truyền hình, phim Hollywood, K-Drama, anime, thể thao trực tiếp. Hỗ trợ 4K HDR.",
        "wifi": "Thiết bị Wi-Fi của FPT sử dụng công nghệ mới nhất Wi-Fi 6/7, hỗ trợ Mesh để mở rộng vùng phủ sóng, phù hợp cho nhà nhiều tầng.",
    }

    product_lower = product.lower()
    for key, desc in descriptions.items():
        if key in product_lower:
            return {"info": desc}

    return {
        "info": f"Sản phẩm '{product}' thuộc danh mục dịch vụ của FPT Telecom. Để biết thêm chi tiết, anh/chị vui lòng cho em biết cụ thể hơn về nhu cầu sử dụng."
    }


def mock_request_agent(**kwargs) -> dict:
    """Mock implementation of request_agent tool."""
    return {
        "info": "Em đã ghi nhận yêu cầu của anh/chị. Nhân viên chăm sóc khách hàng sẽ liên hệ lại trong thời gian sớm nhất (trong vòng 24h làm việc). Anh/chị có thể để lại số điện thoại để được hỗ trợ nhanh hơn."
    }


# Tool registry
TOOL_FUNCTIONS = {
    "get_product_price": mock_get_product_price,
    "describe_product": mock_describe_product,
    "request_agent": mock_request_agent,
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool and return the result."""
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name](**arguments)
    return {"error": f"Unknown tool: {tool_name}"}


# ============================================================================
# TRM Model Inference
# ============================================================================


def predict_with_trm(history: list[dict]) -> tuple[str, Optional[str], dict, dict]:
    """Use TRM model to predict decision and tool, GLiNER2 for entity extraction.

    Returns:
        Tuple of (decision, tool_name, tool_args, slots)
    """
    global model, tokenizer, config, tool_name_to_id
    global gliner2_extractor, tool_param_mapping

    # Build full text from conversation history for GLiNER2
    full_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            full_text += f"{role}: {content}\n"
        elif isinstance(content, dict):
            full_text += f"{role}: {json.dumps(content, ensure_ascii=False)}\n"

    if model is None or tokenizer is None:
        # Mock mode - use GLiNER2 only for extraction
        decision = "direct_answer"
        tool_name = None
        slots, tool_args = {}, {}

        if gliner2_extractor:
            slots, tool_args = gliner2_extractor.extract_all(
                text=full_text,
                tool_name=None,
                tool_params=tool_param_mapping,
            )

        return decision, tool_name, tool_args, slots

    device = next(model.parameters()).device
    id_to_name = {v: k for k, v in tool_name_to_id.items()}

    # Encode conversation
    encoded = tokenizer.encode_conversation_with_offsets(
        history, max_length=config.max_seq_len
    )

    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoded["attention_mask"]], dtype=torch.long, device=device
    )
    role_ids = torch.tensor([encoded["role_ids"]], dtype=torch.long, device=device)

    # Run TRM inference (decision + tool selection only)
    with torch.no_grad():
        outputs = model.inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=role_ids,
        )

    # Get decision from TRM
    decision_prob = torch.sigmoid(outputs.decision_logits[0]).item()
    decision = "tool_call" if decision_prob > 0.5 else "direct_answer"

    tool_name = None
    tool_args = {}
    slots = {}

    if decision == "tool_call":
        # Get tool name from TRM
        tool_idx = outputs.tool_logits[0].argmax().item()
        tool_name = id_to_name.get(tool_idx, f"tool_{tool_idx}")

    # Use GLiNER2 for entity extraction (both slots and tool args)
    if gliner2_extractor:
        slots, tool_args = gliner2_extractor.extract_all(
            text=full_text,
            tool_name=tool_name if decision == "tool_call" else None,
            tool_params=tool_param_mapping,
        )

    return decision, tool_name, tool_args, slots


def build_oss_messages(history: list[dict], after_tool: bool = False) -> list[dict]:
    """Build messages list for OSS API.

    Args:
        history: Conversation history
        after_tool: Whether this is after a tool call

    Returns:
        List of message dicts for OpenAI API
    """
    messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            messages.append({"role": "system", "content": OSS_SYSTEM_PROMPT})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
        elif role == "tool_call":
            # Format tool call for context
            tool_info = f"[Tool Call: {content.get('name', 'unknown')}({json.dumps(content.get('arguments', {}), ensure_ascii=False)})]"
            messages.append({"role": "assistant", "content": tool_info})
        elif role == "tool_response":
            # Format tool response for context
            tool_result = (
                json.dumps(content, ensure_ascii=False)
                if isinstance(content, dict)
                else str(content)
            )
            messages.append(
                {"role": "user", "content": f"[Tool Result: {tool_result}]"}
            )

    # Add instruction to summarize tool result (not call more tools)
    if after_tool:
        messages.append(
            {
                "role": "user",
                "content": "Hãy tổng hợp kết quả tool ở trên và trả lời khách hàng bằng ngôn ngữ tự nhiên. KHÔNG gọi thêm tool.",
            }
        )

    return messages


async def generate_response(history: list[dict], after_tool: bool = False) -> str:
    """Generate response using OpenAI API.

    Args:
        history: Conversation history
        after_tool: Whether this is generating response after a tool call
    """
    global openai_client

    messages = build_oss_messages(history, after_tool)

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        result = response.choices[0].message.content

        # Filter out special tokens if model outputs them
        if result and "<|" in result:
            # Remove special tokens like <|start|>, <|channel|>, etc.
            result = re.sub(r"<\|[^|]+\|>", "", result).strip()
            # If result is empty after filtering, return a fallback
            if not result:
                result = "Dạ, em đã nhận được thông tin. Anh/chị cần em hỗ trợ thêm gì không ạ?"

        return result
    except Exception as e:
        return f"Xin lỗi, em gặp lỗi khi xử lý yêu cầu: {str(e)}"


# ============================================================================
# Chainlit Handlers
# ============================================================================


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    # Load model if not loaded
    if model is None and tokenizer is None:
        load_model()

    # Initialize conversation history
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("history", history)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    history = cl.user_session.get("history", [])

    # Add user message to history
    history.append({"role": "user", "content": message.content})

    # Build full text for GLiNER2
    full_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            full_text += f"{role}: {content}\n"
        elif isinstance(content, dict):
            full_text += f"{role}: {json.dumps(content, ensure_ascii=False)}\n"

    # Step 1: TRM Prediction
    async with cl.Step(name="TRM Prediction", type="tool") as trm_step:
        decision, tool_name, tool_args, slots = predict_with_trm(history)
        trm_step.input = "Analyzing conversation history..."
        trm_output = {
            "decision": decision,
            "tool": tool_name if tool_name else None,
        }
        trm_step.output = json.dumps(trm_output, ensure_ascii=False, indent=2)

    # Step 2: GLiNER2 Entity Extraction
    async with cl.Step(name="GLiNER2 Extraction", type="tool") as gliner_step:
        # Show what labels we're extracting
        labels = list(gliner2_extractor.slot_fields) if gliner2_extractor else []
        if tool_name and tool_name in tool_param_mapping:
            for arg in tool_param_mapping[tool_name]:
                if arg not in labels:
                    labels.append(arg)

        gliner_step.input = json.dumps(
            {
                "text": full_text[-500:] + "..." if len(full_text) > 500 else full_text,
                "labels": labels,
            },
            ensure_ascii=False,
            indent=2,
        )

        gliner_step.output = json.dumps(
            {
                "slots": slots,
                "tool_args": tool_args,
            },
            ensure_ascii=False,
            indent=2,
        )

    if decision == "tool_call" and tool_name:
        # Step 3: Tool Execution
        async with cl.Step(name=f"Tool: {tool_name}", type="tool") as tool_step:
            tool_step.input = json.dumps(tool_args, ensure_ascii=False, indent=2)
            tool_result = execute_tool(tool_name, tool_args)
            tool_step.output = json.dumps(tool_result, ensure_ascii=False, indent=2)

        # Add tool call and response to history
        history.append(
            {
                "role": "tool_call",
                "content": {"name": tool_name, "arguments": tool_args},
            }
        )
        history.append({"role": "tool_response", "content": tool_result})

        # Step 4: LLM Response Generation
        async with cl.Step(name="LLM Generation", type="llm") as llm_step:
            # Build messages preview for input
            messages_preview = build_oss_messages(history, after_tool=True)
            llm_step.input = json.dumps(messages_preview, ensure_ascii=False, indent=2)
            response = await generate_response(history, after_tool=True)
            llm_step.output = response
    else:
        # Step: LLM Response Generation (direct answer)
        async with cl.Step(name="LLM Generation", type="llm") as llm_step:
            messages_preview = build_oss_messages(history, after_tool=False)
            llm_step.input = json.dumps(messages_preview, ensure_ascii=False, indent=2)
            response = await generate_response(history, after_tool=False)
            llm_step.output = response

    # Add assistant response to history
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)

    # Send response
    await cl.Message(content=response).send()


if __name__ == "__main__":
    load_model()
