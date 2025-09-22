# W&B Dashboard Troubleshooting Guide

## 🚨 **ISSUE:** You see only 1 run from an hour ago, but API shows 8 runs

### ✅ **CONFIRMED FACTS:**
- **8 runs are successfully synced** to W&B online
- **API access works correctly** 
- **Runs have proper metadata and tags**
- **Latest test run created successfully** (TEST_RUN_FROM_TERMINAL_1758558534)

---

## 🔧 **TROUBLESHOOTING STEPS:**

### 1. **Browser Cache Issue (Most Likely)**
```bash
# Try these steps in order:
1. Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (Mac)
2. Clear browser cache for wandb.ai
3. Try incognito/private browsing mode
4. Try a different browser
```

### 2. **Check Correct Project URL**
Make sure you're visiting the correct project:
```
✅ CORRECT: https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines
❌ WRONG:   https://wandb.ai/galavny-tel-aviv-university/NLP
```

### 3. **Dashboard Filters**
Check if any filters are applied:
- **Date filters:** Make sure date range includes today
- **Status filters:** Ensure "finished" runs are visible
- **Tag filters:** Clear any tag filters
- **Search filters:** Clear search box

### 4. **Dashboard View Settings**
- Try switching between "Table" and "Cards" view
- Check sorting (try "Created" or "Updated" descending)
- Expand time range if using relative dates

### 5. **Account/Project Access**
- Verify you're logged in as: `galavny`
- Ensure you have access to entity: `galavny-tel-aviv-university`

---

## 🎯 **SPECIFIC TEST:**

Look for this distinctive run I just created:
- **Name:** `TEST_RUN_FROM_TERMINAL_1758558534`
- **ID:** `odhjmoli`
- **Tags:** `DEBUG_TEST`, `TERMINAL_RUN`, `VISIBILITY_CHECK`
- **Direct URL:** https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/odhjmoli

**If you can see this test run, then all 8 runs should be visible!**

---

## 📊 **ALL 8 RUNS THAT SHOULD BE VISIBLE:**

1. **TEST_RUN_FROM_TERMINAL_1758558534** (odhjmoli) - Just created
2. **random_mrpc_seed_123** (hs0gqujf) - MRPC random baseline
3. **random_mrpc_seed_42** (n1w4iza0) - MRPC random baseline  
4. **majority_class_rte** (l7fhwi4q) - RTE majority class
5. **random_sst2_seed_123** (3wkjtzf4) - SST-2 random baseline
6. **random_sst2_seed_42** (u2od0quq) - SST-2 random baseline
7. **majority_class_sst2** (4za274fc) - SST-2 majority class
8. **majority_class_mrpc** (edwnbyzh) - MRPC majority class

---

## 🔄 **ALTERNATIVE ACCESS METHODS:**

### Direct Run URLs:
```
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/odhjmoli
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/hs0gqujf
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/n1w4iza0
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/l7fhwi4q
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/3wkjtzf4
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/u2od0quq
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/4za274fc
https://wandb.ai/galavny-tel-aviv-university/NLP-Baselines/runs/edwnbyzh
```

### Mobile App:
Try the W&B mobile app to see if runs appear there.

---

## 🚨 **IF STILL NO LUCK:**

1. **Check W&B Status Page:** https://status.wandb.ai/
2. **Contact W&B Support:** Could be a platform issue
3. **Use API to export data:**
   ```python
   import wandb
   api = wandb.Api()
   runs = api.runs('galavny-tel-aviv-university/NLP-Baselines')
   for run in runs:
       print(f"{run.name}: {run.summary}")
   ```

---

## ✅ **EXPECTED RESULT:**

After following these steps, you should see **8 runs** in your dashboard showing:
- Multiple baseline types (majority_class, random)
- Multiple tasks (mrpc, sst2, rte)
- Multiple seeds for random baselines
- Proper tags and metadata
- Performance metrics (accuracy, F1, etc.)

**The runs ARE there - it's just a viewing issue!** 🎯
