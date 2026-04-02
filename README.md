# 🤖 The Architecture of Generative AI

> A structured, progressive reference hub for Generative AI — from tokenization to multi-agent systems.  
> Companion project to [The Architecture of Intelligence](../The-Architecture-of-Intelligence) (Deep Learning).

---

## 🗺️ Deep Learning Path

```
01 · Tokenization & Embeddings
        ↓
02 · Language Modeling
        ↓
03 · Transformer LLMs  (GPT → BERT → T5)
        ↓
04 · Pretraining  (objectives, data, scale)
        ↓
05 · Fine-Tuning  (Full FT → LoRA → QLoRA)
        ↓
06 · Alignment  (RLHF → DPO → Constitutional AI)
        ↓
07 · Prompt Engineering  (Zero-shot → CoT → ReAct)
        ↓
08 · RAG  (Naive → Advanced → GraphRAG)
        ↓
09 · Inference Optimization  (Quantization → vLLM)
        ↓
10 · AI Agents  (Tool Use → Planning → Memory)
        ↓
11 · Multi-Agent Systems  (LangGraph, CrewAI)
        ↓
12 · Generative Models  (VAE → GAN → Diffusion)
        ↓
13 · Multimodal AI  (CLIP → VLMs)
        ↓
14 · Architecture Innovations  (MoE → Mamba)
        ↓
15 · Evaluation & Safety
```

---

## 📁 Project Structure

```
The-Architecture-of-GenAI/
│
├── _templates
│   ├── topic_template.py
│   └── _tutorial_template.py
│
├── 00_Basic_ML_Libraries
│   └── topics
│       ├── 00_numpy_library.py
│       ├── 01_scikit_learn_library.py
│       └── 02_tensorflow_library.py
│
├── 01_ML_Foundation_Math_Core
│   └── topics
│       ├── 01_Linear_Algebra.py
│       ├── 02_Calculus_&_Optimization.py
│       └── 03_Statistical_Inference.py
│
├── 02_ML_Foundation_Probability_&_Information
│   └── topics
│       ├── 01_ProbabilisticDistribution.py
│       ├── 02_informationtheory.py
│       └── 03_Probabilistic_Graphical_Models.py
│
├── 03_ML_Foundation_Data
│   └── topics
│       ├── 01_Data_Quality.py
│       ├── 02_DataAugmentation_FeatureEngineering.py
│       └── 03_Sampling_Theory.py
│
├── 04_ML_Foundation_Model_Behaviour
│   └── topics
│       ├── 01_Bias-Variance_Tradeoff.py
│       ├── 02_Overfitting_Underfitting.py
│       ├── 03_Curse_of_Dimensionality.py
│       └── 04_No_Free_Lunch_Theorem.py
│
├── 05_ML_Foundation_Learning_&_Optimization
│   └── topics
│       ├── 00_LossFunctions_Activations.py
│       ├── 01_Optimization_Algorithms.py
│       ├── 02_Learning_Paradigms.py
│       └── 03_Numerical_Stability.py
│
├── 06_ML_Foundation_Evaluation_&_Generalization
│   └── topics
│       ├── 01_Model_Evaluation_Metrics.py
│       └── 02_Regularisation_CrossValidation.py
│
├── 07_ML_Foundation_Advanced_Foundations
│   └── topics
│       ├── 00_Dimensionality_Reduction.py
│       ├── 01_ensemblemethods.py
│       ├── 02_LearningTheory.py
│       ├── 03_Kernel_Methods.py
│       ├── 04_Bayesian_ML.py
│       ├── 05_causality.py
│       ├── 06_Distribution_Shift.py
│       ├── 07_computational_complexity.py
│       ├── 08_inductive_bias.py
│       ├── 09_hyperparameter_optimization.py
│       ├── 10_data_preprocessing.py
│       └── 11_statistical_testing.py
│
├── 08_Supervised_Learning
│   └── topics
│       ├── 00_Supervised_Learning_Path.py
│       ├── 01_regression.py
│       ├── 02_linear_regression.py
│       ├── 03_logistic_regression.py
│       ├── 04_knn.py
│       ├── 05_naive_bayes.py
│       ├── 06_svm.py
│       ├── 07_decision_trees.py
│       ├── 08_ensemble_methods.py
│       ├── 09_random_forests.py
│       └── 10_gradient_boosting.py
│
├── 09_Unsupervised_Learning
│   └── topics
│       ├── 00_Unsupervised_Learning_CoreIdea.py
│       ├── 01_k_means.py
│       ├── 02_dbscan.py
│       ├── 03_hierarchical.py
│       ├── 04_pca.py
│       ├── 05_tsne_umap.py
│       ├── 06_autoencoders.py
│       ├── 07_gmm.py
│       ├── 08_anomaly_detection.py
│       └── 09_neural_network_unsupervised.py
│
├── 10_Deep_Learning
│   └── topics
│       ├── 00_deep_learning_path.py
│       ├── 01_Supervised_Vs_Unsupervised_NN.py
│       ├── 02_Perceptron.py
│       ├── 03_Multilayer_Perceptron.py
│       ├── 04_BackPropagation_Explanation.py
│       ├── 05_ReLU_ActivationFunction.py
│       ├── 06_Convolutional_Neural_Networks.py
│       ├── 07_Recurrent_Neural_Network.py
│       ├── 08_Transformer_Architecture_Model.py
│       ├── 09_Full_Fine_Tuning.py
│       ├── 09b_FT_PEFT_Additive_Breakdown.py
│       ├── 09c_PEFT_Reparameterization_LORA.py
│       ├── 09d_PEFT_Reparameterization_QLORA.py
│       ├── 09e_PEFT_PromptBased_PromptTuning.py
│       └── 10_neural_networks.py
│
├── 11_Training_Core
│   └── topics
│       ├── 00_Order.py
│       ├── 01_weightinit_normalisation.py
│       ├── 02_optimisers_learningratestrategies.py
│       ├── 03_activations_lossfunctions.py
│       ├── 04_backpropagation.py
│       ├── 05_advanceTraining.py
│       ├── 06_Regularisation.py
│       ├── 07_EvaluationDuringTraining.py
│       ├── 08_Gradient_flow_issues.py
│       ├── 09_Data_Pipeline.py
│       ├── 10_Hyperparameter_tuning.py
│       ├── 11_Training_stability.py
│       ├── 12_Checkpointing.py
│       └── 13_train_vs_inference_mode.py
│
├── 12_Reinforcement_Learning
│   └── topics
│       ├── 00_Reinforced_Learning_path.py
│       ├── 01_mdp.py
│       ├── 02_q_learning.py
│       ├── 03_deep_q_network.py
│       ├── 04_policy_gradient.py
│       ├── 05_ppo.py
│       └── 06_actor_critic.py
│
├── 13_Generative_AI
│   └── topics
│       ├── 00_learning_path.py
│       ├── 01_tokenization_embeddings.py
│       ├── 02_language_modeling.py
│       ├── 03_transformer_llms.py
│       ├── 04_pretraining.py
│       ├── 05_fine_tuning.py
│       ├── 06_alignment.py
│       ├── 07_prompt_engineering.py
│       ├── 08_rag.py
│       ├── 09_inference_optimization.py
│       ├── 10_ai_agents.py
│       ├── 11_multi_agent_systems.py
│       ├── 12_generative_models.py
│       ├── 13_multimodal_ai.py
│       ├── 14_architecture_innovations.py
│       └── 15_evaluation_safety.py
│
├── 14_Frameworks
│   └── topics
│       ├── 00_Order.py
│       ├── 00a_AI_Frameworks.py
│       ├── 01_classical_ml_sklearn.py
│       ├── 02_classical_ml_Xgboost_framework.py
│       ├── 03_classical_ml_Lightgbm.py
│       ├── 04_deep_learning_pytorch.py
│       ├── 05_deep_learning_pytorch_lightning.py
│       ├── 06_deep_learning_TensorFlow.py
│       ├── 07_deep_learning_keras.py
│       ├── 08_jax_flax.py
│       ├── 09_nlp_llm_huggingface_transformer.py
│       ├── 09b_nlp_llm_langchain.py
│       ├── 10_nlp_llm_llama_index.py
│       ├── 11_computer_vision_opencv.py
│       ├── 12_computer_vision_rl_gymnasium.py
│       ├── 13_frameworks_distributed_ray.py
│       ├── 14_frameworks_distributed_kubeflow.py
│       ├── 15_frameworks_deployment_onnx.py
│       ├── 16_frameworks_deployment_mlflow.py
│       ├── 17_frameworks_deployment_fastapi.py
│       ├── 18_frameworks_deployment_torchserve.py
│       └── 19_frameworks_mxnet.py
│
├── 15_Interpretability
│   └── topics
│       ├── 00_Order.py
│       ├── 01_scope_and_taxonomy.py
│       ├── 02_Intrinsic_models.py
│       ├── 03_post_hoc_agnostic.py
│       ├── 03a_SHAP.py
│       ├── 03b_lime.py
│       ├── 03c_PDP.py
│       ├── 03d_ALE.py
│       ├── 03e_ICE.py
│       ├── 03f_Anchors.py
│       ├── 04_nn_gradient_based_xai.py
│       ├── 04a_nn_gradient_based_saliency_maps.py
│       ├── 04b_nn_gradient_based_GradCAM.py
│       ├── 04c_nn_gradient_based_DeepLIFT.py
│       ├── 04d_nn_gradient_based_LRP.py
│       ├── 04e_nn_attention_analysis_BERTViz.py
│       ├── 05_nn_mechanistic_interpretability_circuits.py
│       ├── 05a_nn_mechanistic_interpretability_superposition.py
│       ├── 05b_nn_mechanistic_interpretability_Probing_classifiers.py
│       ├── 06_model_calibration.py
│       ├── 07_evaluation_of_explanations.py
│       ├── 08_Fairness_and_Bias.py
│       └── 09_llm_interpretability.py
│
├── 16_Compilers_Runtimes
│   └── topics
│       ├── 00_Order.py
│       ├── 01_Compiler_Infrastructure_LLVM.py
│       ├── 02_Compiler_Infrastructure_MLIR.py
│       ├── 03_Compiler_Infrastructure_CIRCT.py
│       ├── 04_Compiler_Infrastructure_Enzyme.py
│       ├── 05_ML_Compilers_XLA.py
│       ├── 06_ML_Compilers_OpenXLA.py
│       ├── 06a_ML_Compilers_StableHLO.py
│       ├── 07_ML_Compilers_TVM.py
│       ├── 07a_ML_Compilers_TVM_Extra.py
│       ├── 08_ML_Compilers_TorchDynamo.py
│       ├── 09_ML_Compilers_TorchInductor.py
│       ├── 10_Inference_ONNX.py
│       ├── 11_Inference_TensorRT.py
│       ├── 12_Inference_TFLite.py
│       ├── 13_Inference_OpenVINO.py
│       ├── 14_Inference_CoreML.py
│       ├── 14a_Inference_MIGraphX.py
│       ├── 15_Inference_MNN.py
│       ├── 16_Inference_NCNN.py
│       ├── 17_Inference_LLM_vLLM.py
│       ├── 18_Inference_LLM_TensorRT_LLM.py
│       ├── 19_Hardware_CUDA.py
│       ├── 20_Hardware_OpenCL.py
│       ├── 21_GPU_Inference.py
│       └── 22_sglang.py
│
├── 17_GPU_Kernels
│   └── topics
│       ├── 00_order.py
│       ├── 01_foundation_memory_coalescing_shared_mem.py
│       ├── 02_foundation_warp_primitives.py
│       ├── 03_foundation_reduction_kernels.py
│       ├── 04_foundation_prefix_scan.py
│       ├── 05_profiling_benchmarking_NsightCompute.py
│       ├── 06_profiling_benchmarking_NCV_NVTX.py
│       ├── 06a_profiling_benchmarking_NsightSystems.py
│       ├── 07_Dense_Linear_cublas.py
│       ├── 08_Dense_Linear_tensor_cores_wmma.py
│       ├── 09_Dense_Linear_mixed_precision.py
│       ├── 10_Dense_Linear_cuDNN.py
│       ├── 11_Dense_Linear_CUTLASS.py
│       ├── 12_ML_Specific_Flash_Attention.py
│       ├── 13_ML_Specific_Fused_Kernels.py
│       ├── 14_ML_Specific_Quantization_Kernels.py
│       ├── 14a_ML_Specific_Quantization_Persistent_kernels.py
│       ├── 14b_ML_Specific_Quantization_kv_cache_management.py
│       ├── 15_FFT_SPARSE_cufft.py
│       ├── 15a_FFT_SPARSE_cusparse.py
│       ├── 15b_FFT_SPARSE_GEMM_GEMV.py
│       ├── 16_Systems_Tooling_cuda_graphs.py
│       ├── 17_Systems_Tooling_streams_async.py
│       ├── 17a_Systems_Tooling_multi_gpu_memory.py
│       ├── 17b_Multi_GPU_Training.py
│       ├── 18_Systems_Tooling_nccl_collectives.py
│       ├── 19_Systems_Tooling_Triton.py
│       └── 20_Systems_Tooling_ptx_sass.py
│
├── 18_LLM_Training
│   └── topics
│       ├── 00_Order.py
│       ├── 01_mlsystemdesign.py
│       ├── 02_singlegpu_llm_training.py
│       ├── 03_multigpu_llm_training.py
│       └── 04_llm_posttraining_pipeline.py
│
└── 19_Infrastructure_&_MLOps
│    └── topics
│        ├── 01_Linux_Path.py
│        ├── 02_docker_fundamentals.py
│        └── 03_kubernetes_fundamentals.py
│
│
└── README.md
```

---

## 🚀 Running the App

```bash
# Install dependencies
pip install streamlit transformers torch gensim

# Run
streamlit run app.py
```

---

## 📐 Adding a New Topic

1. Copy `topics/topic_template.py`
2. Rename it with a numeric prefix (e.g. `16_new_topic.py`)
3. Fill in `THEORY`, `OPERATIONS`, and `visual_html`
4. Add a visual file to `Required_Images/new_topic_visual.py`
5. The app auto-discovers it — no registration needed

---

## 🔗 Connection to Architecture of Intelligence

| Deep Learning Project | Gen AI Project                  |
|----------------------|----------------------------------|
| Perceptron → MLP     | Tokenization → Language Modeling |
| CNN, RNN             | Transformer LLM families         |
| Transformer          | Pretraining paradigms            |
| Full Fine-Tuning     | Instruction Fine-Tuning          |
| LoRA / QLoRA         | LoRA / QLoRA (extended)          |
| —                    | Alignment (RLHF, DPO)            |
| —                    | RAG, Agents, Multi-Agent         |
| —                    | Diffusion, Multimodal            |

---

*Built as a structured learning reference — theory, visuals, and runnable code for every concept.*
