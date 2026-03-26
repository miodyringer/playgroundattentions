from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from transformer_lens import utils, HookedTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Model Initialization
model_name = os.getenv("MODEL_NAME")
compare_model_name = os.getenv("COMPARE_MODEL")

# Only Mistral has Gated Access. HF_TOKEN is required. Get it on https://huggingface.co/settings/tokens
if "mistral" in model_name and os.getenv("HF_TOKEN") is None:
        raise ValueError("Mistral models require HF_TOKEN environment variable to be set in .env file")

device = utils.get_device()
model = HookedTransformer.from_pretrained(model_name, device=device)

# Load comparison model if specified
compare_model = None
if compare_model_name:
    if "mistral" in compare_model_name and os.getenv("HF_TOKEN") is None:
        raise ValueError("Mistral models require HF_TOKEN environment variable to be set in .env file")
    compare_model = HookedTransformer.from_pretrained(compare_model_name, device=device)


app = FastAPI(title="Attention Capture API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100


class GenerateResponse(BaseModel):
    answer: str
    model: str
    metadata: Dict[str, Any]


class AnalyzeRequest(BaseModel):
    answer: str
    attn_layer: Optional[int] = -1


class AnalyzeResponse(BaseModel):
    attention_pattern: list
    shape: list[int]
    num_tokens: int
    tokens: list[str]


@app.get("/")
async def root():
    return {"message": "LLM API is running"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    try:
        # Tokenize the prompt
        tokens = model.to_tokens(request.prompt)
        prompt_length = tokens.shape[1]

        # Generate tokens
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_at_eos=True
        )

        # Calculate actual new tokens generated
        total_length = generated_tokens.shape[1]
        actual_new_tokens = total_length - prompt_length

        # Decode the full generated sequence
        full_answer = model.to_string(generated_tokens[0])

        # Extract only the new tokens (everything after the prompt)
        new_tokens_only = generated_tokens[0][prompt_length:]
        new_answer = model.to_string(new_tokens_only)

        return GenerateResponse(
            answer=full_answer,
            model=model_name,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "prompt_tokens": int(prompt_length),
                "generated_tokens": int(actual_new_tokens),
                "new_text": new_answer,
                "total_tokens": int(total_length)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_answer(request: AnalyzeRequest):
    try:
        import torch

        # Tokenization
        tokens = model.to_tokens(request.answer)
        num_tokens = len(tokens[0])  # Remove batch dimension for count

        # All layers
        if request.attn_layer == -1:
            # Generate hook names for all layers
            num_layers = model.cfg.n_layers
            attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(num_layers)]

            _, attn_cache = model.run_with_cache(
                tokens,
                remove_batch_dim=True,
                names_filter=attn_hook_names
            )
        # Layers 0 to n
        else:
            # Generate hook names for layers 0 through attn_layer
            attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(request.attn_layer + 1)]

            _, attn_cache = model.run_with_cache(
                tokens,
                remove_batch_dim=True,
                stop_at_layer=request.attn_layer + 1,
                names_filter=attn_hook_names
            )

        # Stack all requested layers: [num_layers, num_heads, seq_len, seq_len]
        attn = torch.stack([attn_cache[hook_name] for hook_name in attn_hook_names])

        # Convert tensor to list for JSON serialization
        attn_list = attn.tolist()
        shape = list(attn.shape)

        # Get tokens as strings
        token_strings = [model.to_string(tokens[0][i]) for i in range(num_tokens)]

        return AnalyzeResponse(
            attention_pattern=attn_list,
            shape=shape,
            num_tokens=num_tokens,
            tokens=token_strings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=AnalyzeResponse)
async def compare_analyze(request: AnalyzeRequest):
    """Analyze text using the comparison model"""
    if not compare_model:
        raise HTTPException(status_code=400, detail="No comparison model configured. Set COMPARE_MODEL in .env file")

    try:
        import torch

        # Tokenization with comparison model
        tokens = compare_model.to_tokens(request.answer)
        num_tokens = len(tokens[0])

        # All layers
        if request.attn_layer == -1:
            num_layers = compare_model.cfg.n_layers
            attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(num_layers)]

            _, attn_cache = compare_model.run_with_cache(
                tokens,
                remove_batch_dim=True,
                names_filter=attn_hook_names
            )
        # Layers 0 to n
        else:
            attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(request.attn_layer + 1)]

            _, attn_cache = compare_model.run_with_cache(
                tokens,
                remove_batch_dim=True,
                stop_at_layer=request.attn_layer + 1,
                names_filter=attn_hook_names
            )

        # Stack all requested layers
        attn = torch.stack([attn_cache[hook_name] for hook_name in attn_hook_names])

        # Convert tensor to list for JSON serialization
        attn_list = attn.tolist()
        shape = list(attn.shape)

        # Get tokens as strings
        token_strings = [compare_model.to_string(tokens[0][i]) for i in range(num_tokens)]

        return AnalyzeResponse(
            attention_pattern=attn_list,
            shape=shape,
            num_tokens=num_tokens,
            tokens=token_strings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/models")
async def get_models():
    """Return the primary and comparison model names"""
    return {
        "primary_model": model_name,
        "compare_model": compare_model_name
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
