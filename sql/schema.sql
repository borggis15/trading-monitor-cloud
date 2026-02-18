create table if not exists bars_15m (
  exchange text not null,
  symbol   text not null,
  ts       timestamptz not null,
  open     double precision,
  high     double precision,
  low      double precision,
  close    double precision,
  volume   double precision,
  source   text,
  primary key (exchange, symbol, ts)
);

create table if not exists features_15m (
  exchange text not null,
  symbol   text not null,
  ts       timestamptz not null,
  rsi      double precision,
  ema_fast double precision,
  ema_slow double precision,
  atr      double precision,
  zscore   double precision,
  ret_fwd  double precision,
  primary key (exchange, symbol, ts)
);

create table if not exists model_registry (
  model_id text primary key,
  created_at timestamptz not null default now(),
  exchange text not null,
  horizon_bars int not null,
  model_type text not null,
  train_rows int,
  notes text
);

create table if not exists signals (
  exchange text not null,
  symbol   text not null,
  ts       timestamptz not null,
  action   text not null,
  proba_up double precision,
  ret_exp  double precision,
  risk_est double precision,
  ev_bps   double precision,
  size_eur double precision,
  sl_price double precision,
  tp_price double precision,
  horizon text,
  explanation text,
  model_id text,
  primary key (exchange, symbol, ts)
);

create table if not exists backtest_summary (
  exchange text not null,
  symbol text not null,
  run_at timestamptz not null default now(),
  sharpe double precision,
  sortino double precision,
  max_drawdown double precision,
  win_rate double precision,
  profit_factor double precision,
  trades int,
  primary key (exchange, symbol)
);
