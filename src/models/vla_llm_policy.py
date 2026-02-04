import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class VLAPolicy(nn.Module):
    def __init__(self, model_name, state_dim, action_dim, load_in_4bit=False):
        super().__init__()
        
        print(f"Loading VLA Backbone: {model_name}...")
        
        # 1. Load LLM Backbone
        # Use bfloat16 for V100/A100 (better stability than float16)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            load_in_4bit=load_in_4bit
        )
        
        # 2. Apply LoRA (Make it trainable efficiently)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=32, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all Linear layers for max expressivity
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        # 3. Projectors
        # State Projector: Projects Robot State (Dim X) -> LLM Embedding Dim (Dim 4096 for Llama 3)
        self.hidden_size = self.llm.config.hidden_size
        
        self.state_projector = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(dtype) # Match LLM dtype
        
        # Action Head: Projects Last Token Hidden State -> Action Dim
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, action_dim)
        ).to(dtype)

    def forward(self, state, text_list):
        """
        state: (Batch, State_Dim)
        text_list: List[str] of instructions e.g. ["Lift cube", "Push nut"]
        """
        
        # 1. Embed Robot State
        # (Batch, 1, Hidden) -> represents the "Visual/State" token
        state_emb = self.state_projector(state).unsqueeze(1) 
        
        # 2. Tokenize Text
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        ).to(self.llm.device)
        
        # 3. Get Text Embeddings from LLM
        # We need the raw embeddings to concat with state_emb
        input_embeds = self.llm.get_input_embeddings()(inputs.input_ids) # (Batch, Seq, Hidden)
        
        # 4. Concatenate: [State_Token, Text_Tokens]
        # This is the "Multimodal" sequence
        combined_embeds = torch.cat([state_emb, input_embeds], dim=1)
        
        # Update Attention Mask for the new state token
        # Add column of 1s to the left
        batch_size = state.shape[0]
        ones = torch.ones((batch_size, 1), device=self.llm.device)
        combined_mask = torch.cat([ones, inputs.attention_mask], dim=1)
        
        # 5. Pass through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        
        # 6. Predict Action from LAST token
        # Why last? Because it has attended to State + "Instruction"
        last_hidden_state = outputs.hidden_states[-1] # (Batch, Seq+1, Hidden)
        
        # Extract the vector corresponding to the last real token
        # (We use generalized pooling or just take the last index)
        # Simple/Fast: Take the last element of the sequence
        last_token_emb = last_hidden_state[:, -1, :] 
        
        action_pred = self.action_head(last_token_emb)
        
        return action_pred
