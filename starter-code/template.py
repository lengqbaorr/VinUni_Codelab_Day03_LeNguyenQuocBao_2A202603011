"""
Lab #3: Baseline Chatbot vs ReAct Agent.

The implementation is deterministic so the lab can demonstrate the ReAct
control flow without requiring an API key or an external LLM service.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional

from tools import TOOL_DEFINITIONS, TOOL_MAP


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline chatbot: answer once and never call a tool."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "answer": (
                f"[Chatbot Baseline] Không thể phản hồi yêu cầu: {user_input}"
            ),
            "tool_calls": [],
        }


class ReActAgent:
    """A small Thought-Action-Observation agent backed by the local tools."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max(0, max_iterations)
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _parse_budget(user_input: str) -> int:
        """Convert common Vietnamese price expressions to an integer VND value."""
        text = user_input.lower().replace(",", ".")

        million = re.search(r"(\d+(?:\.\d+)?)\s*(?:triệu|trieu)", text)
        if million:
            return int(float(million.group(1)) * 1_000_000)

        thousand = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
        if thousand:
            return int(float(thousand.group(1)) * 1_000)

        raw_vnd = re.search(r"(\d[\d.]*)\s*(?:vnd|đ|đồng|dong)\b", text)
        if raw_vnd:
            return int(raw_vnd.group(1).replace(".", ""))

        return 5_000_000

    @staticmethod
    def _extract_route(user_input: str) -> tuple[Optional[str], Optional[str]]:
        codes = re.findall(r"\b(?:HAN|SGN|DAD)\b", user_input.upper())
        if len(codes) >= 2:
            return codes[0], codes[1]
        return None, None

    @staticmethod
    def _extract_city(user_input: str) -> Optional[str]:
        codes = re.findall(r"\b(?:HAN|SGN|DAD)\b", user_input.upper())
        if codes:
            return codes[-1]

        text = user_input.lower()
        city_aliases = {
            "hà nội": "HAN",
            "ha noi": "HAN",
            "đà nẵng": "DAD",
            "da nang": "DAD",
            "hồ chí minh": "SGN",
            "ho chi minh": "SGN",
            "sài gòn": "SGN",
            "sai gon": "SGN",
        }
        return next((code for name, code in city_aliases.items() if name in text), None)

    def _execute_tool(
        self, name: str, args: Dict[str, Any], thought: str
    ) -> Any:
        """Safely call a registered tool and append one complete trace entry."""
        normalized_name = name.strip().lower()
        tool = TOOL_MAP.get(normalized_name)

        if tool is None:
            observation: Any = {"error": f"Unknown tool: {normalized_name}"}
        else:
            try:
                observation = tool(**args)
            except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                observation = {"error": str(exc)}

        self.trace.append(
            {
                "iteration": len(self.trace) + 1,
                "thought": thought,
                "action": {"name": normalized_name, "args": args},
                "observation": observation,
            }
        )
        return observation

    def _completed(self, answer: str) -> Dict[str, Any]:
        return {
            "status": "completed",
            "iterations": len(self.trace),
            "answer": answer,
            "trace": self.trace,
        }

    def _max_iterations_result(self) -> Dict[str, Any]:
        return {
            "status": "max_iterations_reached",
            "iterations": len(self.trace),
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
        }

    @staticmethod
    def _format_flights(flights: List[Dict[str, Any]]) -> str:
        if not flights:
            return "Không tìm thấy chuyến bay phù hợp với hành trình và ngân sách."

        details = []
        for flight in flights:
            details.append(
                f"{flight['flight_number']} ({flight['airline']}), khởi hành "
                f"{flight['departure_time']}, giá {flight['price_vnd']:,} VND"
            )
        return "Các chuyến bay phù hợp: " + "; ".join(details) + "."

    @staticmethod
    def _format_weather(weather: Dict[str, Any]) -> str:
        if "error" in weather:
            return f"Không lấy được thông tin thời tiết: {weather['error']}."
        return (
            f"Thời tiết tại {weather['city']}: {weather['temperature_c']}°C, "
            f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
            f"Gợi ý: {weather['recommendation']}"
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        # A ReAct trace belongs to exactly one request.
        self.trace = []
        text = user_input.lower()
        needs_weather = any(
            phrase in text
            for phrase in ("thời tiết", "thoi tiet", "nhiệt độ", "nhiet do", "mặc gì")
        )
        flight_search_cues = ("tìm", "tim", "đặt", "dat", "giá", "gia", "dưới", "duoi")
        needs_flight = (
            "chuyến bay" in text
            or "chuyen bay" in text
            or "bay từ" in text
            or (
                any(phrase in text for phrase in ("vé", "ve may bay"))
                and any(cue in text for cue in flight_search_cues)
                and not any(phrase in text for phrase in ("chính sách", "chinh sach", "đổi trả", "doi tra"))
            )
        )

        if self.max_iterations == 0:
            return self._max_iterations_result()

        # Questions outside the supported live-data intents need no tool call.
        if not needs_flight and not needs_weather:
            self.trace.append(
                {
                    "iteration": 1,
                    "thought": "Đây là câu hỏi FAQ, không cần gọi công cụ dữ liệu.",
                    "action": None,
                    "observation": "Trả lời bằng thông tin hướng dẫn chung.",
                }
            )
            if "vinpearl" in text and any(word in text for word in ("đổi", "trả", "hoàn")):
                answer = (
                    "Chính sách đổi, hoàn vé Vinpearl phụ thuộc vào điều kiện của "
                    "hạng vé và nhà vận chuyển. Bạn nên kiểm tra điều kiện trên xác "
                    "nhận đặt chỗ hoặc liên hệ bộ phận hỗ trợ Vinpearl để được xác nhận."
                )
            else:
                answer = "Tôi đã nhận được câu hỏi của bạn và sẽ hỗ trợ trong phạm vi thông tin hiện có."
            return self._completed(answer)

        flight_answer = ""
        if needs_flight:
            origin, destination = self._extract_route(user_input)
            if not origin or not destination:
                self.trace.append(
                    {
                        "iteration": 1,
                        "thought": "Yêu cầu tìm chuyến bay còn thiếu hành trình.",
                        "action": None,
                        "observation": {"error": "Missing origin or destination"},
                    }
                )
                return self._completed(
                    "Vui lòng cung cấp mã sân bay đi và đến, ví dụ HAN đi SGN."
                )

            flights = self._execute_tool(
                "get_flight_info",
                {
                    "origin": origin,
                    "destination": destination,
                    "max_price": self._parse_budget(user_input),
                },
                "Cần tra cứu các chuyến bay phù hợp với hành trình và ngân sách.",
            )
            flight_answer = self._format_flights(flights)

            if not needs_weather:
                return self._completed(flight_answer)

        if len(self.trace) >= self.max_iterations:
            return self._max_iterations_result()

        weather_answer = ""
        if needs_weather:
            city_code = self._extract_city(user_input)
            if not city_code:
                self.trace.append(
                    {
                        "iteration": len(self.trace) + 1,
                        "thought": "Yêu cầu thời tiết chưa xác định được thành phố.",
                        "action": None,
                        "observation": {"error": "Missing city code"},
                    }
                )
                return self._completed("Vui lòng cho biết thành phố cần xem thời tiết.")

            weather = self._execute_tool(
                "get_weather_forecast",
                {"city_code": city_code},
                "Cần tra cứu thời tiết và gợi ý trang phục tại điểm đến.",
            )
            weather_answer = self._format_weather(weather)

            if not needs_flight:
                return self._completed(weather_answer)

        # A combined request uses a third trace step to explicitly synthesize the
        # two observations, matching the Thought-Action-Observation lab diagram.
        if len(self.trace) >= self.max_iterations:
            return self._max_iterations_result()

        answer = f"{flight_answer} {weather_answer}"
        self.trace.append(
            {
                "iteration": len(self.trace) + 1,
                "thought": "Đã đủ dữ liệu chuyến bay và thời tiết để tổng hợp câu trả lời.",
                "action": None,
                "observation": "Final Answer",
            }
        )
        return self._completed(answer)


def main() -> None:
    # Keep the Vietnamese demo readable in Windows terminals using legacy encodings.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = (
        "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, "
        "rồi cho biết thời tiết SGN nên mặc gì?"
    )

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
