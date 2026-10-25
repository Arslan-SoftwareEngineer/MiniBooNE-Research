export interface AppConfig { apiBase: string; }

export interface FeatureInfo {
  name: string; min: number; max: number; mean: number; std: number;
  importance: number; rank: number;
}
export interface Sample { label: string; y: number; values: number[]; }
export interface ModelInfo {
  model_id: string; synthetic: boolean; version: string;
  classes: Record<string, string>; n_train: number; n_test: number;
  features: FeatureInfo[]; samples: Sample[];
}

export interface Series { mean: number[]; std: number[]; }
export interface Results {
  model_id: string; synthetic: boolean; source: string;
  n_rows: number; n_features: number; n_train: number; n_test: number; split: string;
  replication: { model: string; accuracy: number; auc: number }[];
  dropout: { rates: number[]; seeds: number; series: Record<string, Series> };
  noise: {
    alphas: number[]; seeds: number; k: number;
    groups: Record<string, { columns: number[]; series: Record<string, Series>;
                             collapse_alpha: Record<string, number | null> }>;
  };
  top_features: string[]; bottom_features: string[];
}

export interface Reading {
  prediction: number; label: string; probability_signal: number; probabilities: number[];
}
export interface CompareResponse {
  models: Record<string, { intact: Reading; degraded: Reading }>;
  offline: string[]; n_offline: number; model_id: string; disclaimer: string;
}
export interface Health {
  status: string; model_loaded: boolean; features: number; models: string[];
  model_id: string | null; synthetic: boolean | null;
}

export type ApiState = 'checking' | 'online' | 'offline' | 'static';

export const MODEL_STYLE: Record<string, { label: string; color: string }> = {
  xgb_baseline: { label: 'XGBoost baseline', color: '#B4512B' },
  xgb_robust: { label: 'XGBoost, dropout-trained', color: '#0A7B83' },
  xgb_baseline_mean_impute: { label: 'Baseline + mean-fill (knows which sensors died)', color: '#7C8794' },
  mlp: { label: 'MLP', color: '#6A5BC2' },
};
