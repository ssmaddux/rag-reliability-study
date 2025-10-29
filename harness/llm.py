"""
LLM backends:
- dummy: returns templated answers using retrieved context (deterministic-ish)
- openai: (stub) shows how you'd call OpenAI; fill in your key and install openai
- llama_cpp: (stub) shows how to run a local GGUF model

Start with 'dummy' for teaching; switch later when you're ready.
"""

import os, hashlib, random
from typing import Optional

class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.llm.backend
        self.temperature = float(cfg.llm.temperature)
        self.top_p = float(cfg.llm.top_p)
        self.seed = int(getattr(cfg.llm, "seed", 1234))

        if self.backend == "openai":
            try:
                import openai  # type: ignore
                self.openai = openai
            except Exception as e:
                raise RuntimeError("Install openai to use backend=openai") from e
        elif self.backend == "llama_cpp":
            try:
                from llama_cpp import Llama  # type: ignore
                model_path = cfg.llm.model_path or ""
                if not model_path:
                    raise RuntimeError("Set llm.model_path to your GGUF file for llama_cpp backend.")
                # Load a modest context; adjust n_ctx as needed
                self.llama = Llama(model_path=model_path, n_ctx=4096, seed=self.seed)
            except Exception as e:
                raise RuntimeError("Install llama-cpp-python and set a valid model_path.") from e

    def generate(self, prompt: str, context: str) -> str:
        # Dummy backend: deterministic-ish templated answers using provided context
        if self.backend == "dummy":
            random.seed(hash((prompt, self.seed)) % (2**32))
            variants = [
                "Here's what I found: ",
                "According to university policy: ",
                "Based on the knowledge base: ",
            ]
            prefix = variants[random.randint(0, len(variants) - 1)]

            # Extract first cited article ID as a "grounding" reference
            anchor = ""
            for line in context.splitlines():
                if line.startswith("[KA-"):
                    anchor = line.split("]")[0].strip("[]")
                    break

            pl = prompt.lower()
            if "drop a course" in pl:
                answer = "Before the census date, drop in the portal under Enrollment > Manage Classes. After the deadline, submit a Late Drop Petition."
            elif "student loans" in pl:
                answer = "Complete the FAFSA at studentaid.gov by March 1 and review any verification tasks in the portal."
            elif "schedule" in pl:
                answer = "Open the portal and navigate to Enrollment > My Schedule. Use 'Export to Calendar' for convenience."
            elif "transcripts" in pl:
                answer = "Order via Registrar > Transcripts. Electronic delivery is usually within 24 hours."
            elif "reset" in pl or "locked out" in pl:
                answer = "Use portal.university.edu/reset. If locked out or 2FA is unavailable, contact IT support."
            elif "full-time" in pl:
                answer = "Undergraduates: 12+ credits; Graduates: 9+ credits. Check aid/housing requirements."
            elif "health insurance" in pl:
                answer = "Full-time students are enrolled by default; submit a waiver with proof of coverage by the deadline."
            elif "id card" in pl:
                answer = "New IDs at Orientation; replacements at the Campus Card Office for a $25 fee."
            elif "leave of absence" in pl:
                answer = "Meet with your advisor and submit a Leave Request in the portal. Consider aid and housing impacts."
            elif "appeal a grade" in pl:
                answer = "Contact your instructor within 10 business days; if unresolved, submit a Grade Appeal to the department chair."
            elif "add a class" in pl:
                answer = "Use Enrollment > Add Classes. If the class is full, join the waitlist."
            else:
                answer = "Please check the portal for the relevant section and follow the on-screen steps."

            return f"{prefix}{answer} (Source: {anchor})"

        # OpenAI backend: requires OPENAI_API_KEY and the OpenAI client library
        elif self.backend == "openai":
            from openai import OpenAI
            client = OpenAI()

            system = (
                "You are a helpful university help center assistant. "
                "Use ONLY the provided context; do not invent facts. "
                "If the answer is not in the context, say you don't know. "
                "Cite the article you used in parentheses like (Source: KA-XXXXX)."
            )
            user = f"Context:\n{context}\n\nUser question: {prompt}"

            resp = client.chat.completions.create(
                model=getattr(self.cfg.llm, "model_name", None),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=int(getattr(self.cfg.llm, "max_completion_tokens", 512)),
            )

            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                return getattr(resp, "content", str(resp)).strip()

        # llama.cpp backend: local inference via llama-cpp-python
        elif self.backend == "llama_cpp":
            system = "You are a helpful university help center assistant."
            user = f"Context:\n{context}\n\nUser question: {prompt}"
            out = self.llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )

            try:
                return out["choices"][0]["message"]["content"].strip()
            except Exception:
                return str(out)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")
