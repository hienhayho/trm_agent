"""
Chainlit Chat Application for TRM Agent.

A chat interface that uses TRM model for decision making and tool calling,
with OpenAI for content generation.

Usage:
    # Install chainlit first:
    uv add chainlit openai

    # Run the app:
    uv run chainlit run app.py

    # With custom model checkpoint:
    TRM_CHECKPOINT=outputs/checkpoint.pt uv run chainlit run app.py
"""

import json
import os
from pathlib import Path
from typing import Optional

import chainlit as cl
import torch
from openai import OpenAI

from trm_agent.data import TRMTokenizer
from trm_agent.models import TRMConfig, TRMForToolCalling

# Configuration
CHECKPOINT_PATH = os.environ.get("TRM_CHECKPOINT", "outputs/checkpoint_best.pt")
TOKENIZER_PATH = os.environ.get("TRM_TOKENIZER", "outputs/tokenizer/tokenizer.model")
TOOLS_PATH = os.environ.get("TRM_TOOLS", "data/tools.json")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")

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


def load_model():
    """Load TRM model and tokenizer."""
    global model, tokenizer, config, tools, tool_name_to_id, openai_client

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


def predict_with_trm(history: list[dict]) -> tuple[str, Optional[str], dict]:
    """Use TRM model to predict decision and tool.

    Returns:
        Tuple of (decision, tool_name, tool_args)
    """
    global model, tokenizer, config, tool_name_to_id

    if model is None or tokenizer is None:
        # Mock mode - default to direct_answer
        return "direct_answer", None, {}

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

    # Run inference
    with torch.no_grad():
        outputs = model.inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=role_ids,
        )

    # Get decision
    decision_prob = torch.sigmoid(outputs.decision_logits[0]).item()
    decision = "tool_call" if decision_prob > 0.5 else "direct_answer"

    tool_name = None
    tool_args = {}

    if decision == "tool_call":
        # Get tool name
        tool_idx = outputs.tool_logits[0].argmax().item()
        tool_name = id_to_name.get(tool_idx, f"tool_{tool_idx}")

        # Extract tool arguments from spans
        token_offsets = encoded["offsets"]
        full_text = encoded["full_text"]

        # Get param fields from config
        unified_fields = config.get_unified_fields()
        num_slots = config.num_slots

        # Only extract tool params (not slots)
        for param_idx in range(config.num_tool_params):
            unified_idx = num_slots + param_idx
            param_name = (
                unified_fields[unified_idx]
                if unified_idx < len(unified_fields)
                else None
            )

            if param_name:
                start_pos = (
                    outputs.param_start_logits[0, :, unified_idx].argmax().item()
                )
                end_pos = outputs.param_end_logits[0, :, unified_idx].argmax().item()

                if start_pos < len(token_offsets) and end_pos < len(token_offsets):
                    char_start = token_offsets[start_pos][0]
                    char_end = token_offsets[end_pos][1]
                    if char_start >= 0 and char_end > char_start:
                        arg_value = full_text[char_start:char_end].strip()
                        if arg_value:
                            tool_args[param_name] = arg_value

    return decision, tool_name, tool_args


async def generate_response(history: list[dict]) -> str:
    """Generate response using OpenAI API."""
    global openai_client

    # Convert history to OpenAI format
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

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content
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

    # Get TRM prediction
    decision, tool_name, tool_args = predict_with_trm(history)

    # Show TRM prediction in UI
    trm_info = f"🤖 **TRM Prediction**\n- Decision: `{decision}`"
    if decision == "tool_call" and tool_name:
        trm_info += f"\n- Tool: `{tool_name}`"
        if tool_args:
            trm_info += f"\n- Args: `{json.dumps(tool_args, ensure_ascii=False)}`"
    await cl.Message(content=trm_info).send()

    if decision == "tool_call" and tool_name:
        # Execute tool
        tool_result = execute_tool(tool_name, tool_args)

        # Add tool call and response to history
        history.append(
            {
                "role": "tool_call",
                "content": {"name": tool_name, "arguments": tool_args},
            }
        )
        history.append({"role": "tool_response", "content": tool_result})

        # Show tool execution to user
        tool_msg = cl.Message(
            content=f"🔧 **Đang gọi công cụ**: `{tool_name}`\n**Tham số**: `{json.dumps(tool_args, ensure_ascii=False)}`",
        )
        await tool_msg.send()

        # Generate response based on tool result
        response = await generate_response(history)
    else:
        # Direct answer - generate response
        response = await generate_response(history)

    # Add assistant response to history
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)

    # Send response
    await cl.Message(content=response).send()


if __name__ == "__main__":
    load_model()
