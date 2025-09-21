#!/usr/bin/env python3
"""
Test script to verify wandb connection to galavny-tel-aviv-university/NLP project
"""

import wandb
import random
import time

def test_wandb_connection():
    """Test wandb connection and logging to the specified project"""
    
    # Initialize wandb with the specific project
    wandb.init(
        project="NLP",
        entity="galavny-tel-aviv-university",
        name="connection-test",
        tags=["test", "connection-check"]
    )
    
    print("✅ Successfully connected to wandb!")
    print(f"📊 Project: {wandb.run.project}")
    print(f"🏢 Entity: {wandb.run.entity}")
    print(f"🆔 Run ID: {wandb.run.id}")
    print(f"🔗 Run URL: {wandb.run.url}")
    
    # Log some test metrics
    for step in range(10):
        # Simulate some training metrics
        loss = 1.0 - (step * 0.08) + random.uniform(-0.05, 0.05)
        accuracy = step * 0.09 + random.uniform(-0.02, 0.02)
        
        wandb.log({
            "test_loss": loss,
            "test_accuracy": accuracy,
            "step": step
        })
        
        print(f"Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        time.sleep(0.5)  # Brief pause to simulate training time
    
    # Log a simple config
    wandb.config.update({
        "test_parameter": "connection_test",
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "test_model"
    })
    
    print("📈 Test metrics logged successfully!")
    print("🎯 Test completed - check your wandb dashboard!")
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    try:
        test_wandb_connection()
        print("\n🎉 wandb connection test PASSED!")
        print("You can view the results at: https://wandb.ai/galavny-tel-aviv-university/NLP")
        
    except Exception as e:
        print(f"\n❌ wandb connection test FAILED!")
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're logged in: wandb login")
        print("2. Check your API key is valid")
        print("3. Verify you have access to the galavny-tel-aviv-university entity")
        print("4. Ensure wandb is installed: pip install wandb")
