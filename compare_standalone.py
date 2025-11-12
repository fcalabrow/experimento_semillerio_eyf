#!/usr/bin/env python3
"""
Librerías necesarias (instalar con pip):
- polars>=0.19.0
- lightgbm>=4.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- pyyaml>=6.0
- scikit-learn>=1.3.0
- psutil>=5.9.0
- google-cloud-storage>=2.10.0
"""

import os
import yaml
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb
from zoneinfo import ZoneInfo
import polars as pl
import psutil
from google.cloud import storage 

# ============================================================================
# VARIABLES DE CONFIGURACIÓN
# ============================================================================

# URL del dataset en GCP bucket
DATASET_GCP_URL = "gs://dosdesvios_bukito3/competencia_02/data/02_v2.parquet"

# Meses de validación
VAL_MONTH = [202104]

# Configuración del experimento
EXPERIMENT_NAME = "semillerio_vs_no_semillerio_abril"
N_EXPERIMENTS = 30

# Seeds base
SEEDS_BASE = [0, 9, 12, 20, 18]

# Parámetros fijos
FIXED_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbose": 1,
    "force_row_wise": True,
    "boost_from_average": True,
    "feature_pre_filter": True,
    "max_bin": 31,
}

# Ganancias
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000

# Configuración para la meseta
BASE_PLATEAU_WINDOW = 2000

# Configuración de modelos a comparar
MODELS_CONFIG = {
    "model_sin_semillerio": {
        "params": {
            "learning_rate": 0.022,
            "feature_fraction": 0.43,
            "min_data_in_leaf": 433,
            "bagging_freq": 4,
            "neg_bagging_fraction": 0.55,
            "pos_bagging_fraction": 0.52,
            "num_boost_round": 458,
            "num_leaves": 13,
        },
        "semillerio": 1,
        "n_submissions": 11000,
        "months": [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201911, 201912, 202001, 202002, 202003, 202004, 202005, 202007, 202008, 202009, 202010, 202011, 202012, 202101, 202102],
        "chosen_features": ["seleccion_730"],
        "undersampling_fraction": 0.05,
        "undersampling_stratified": False,
    },
    "model_con_semillerio": {
        "params": {
            "learning_rate": 0.38,
            "feature_fraction": 0.78,
            "min_data_in_leaf": 2222,
            "num_boost_round": 20,
            "num_leaves": 30,
        },
        "semillerio": 100,
        "n_submissions": 11000,
        "months": [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201911, 201912, 202001, 202002, 202003, 202004, 202005, 202007, 202008, 202009, 202010, 202011, 202012, 202101, 202102],
        "chosen_features": ["seleccion_730"],
        "undersampling_fraction": 0.05,
        "undersampling_stratified": False,
    },
}

FEATURE_SUBSETS = {
    "seleccion_730": [
        'ctrx_quarter_normalizado', 'rankp_mcaja_ahorro', 'ctrx_quarter', 'rankn_mcaja_ahorro',
        'rankp_ctarjeta_visa_transacciones', 'rankp_ctrx_quarter', 'rankn_ctrx_quarter',
        'cpayroll_trx', 'mcaja_ahorro', 'ctarjeta_visa_transacciones', 'mpayroll_sobre_edad',
        'mtarjeta_visa_consumo', 'rankp_ctarjeta_visa', 'rankn_mtarjeta_visa_consumo',
        'rankp_mtarjeta_visa_consumo', 'mtarjeta_visa_consumo_ratioavg6', 'rankn_mpasivos_margen',
        'ctarjeta_visa_ratioavg6', 'rankp_mcuentas_saldo', 'mcaja_ahorro_lag1', 'ctrx_quarter_lag1',
        'mcaja_ahorro_min6', 'cproductos_ratioavg6', 'mpayroll', 'rankp_mpasivos_margen',
        'rankp_Visa_msaldopesos', 'rankn_mcuentas_saldo', 'ctrx_quarter_normalizado_lag1',
        'mcuentas_saldo', 'rankn_mcuenta_corriente', 'ctrx_quarter_ratioavg6',
        'ctarjeta_visa_transacciones_ratioavg6', 'cdescubierto_preacordado_ratioavg6',
        'rankp_mcuenta_corriente', 'Visa_mconsumototal', 'Visa_mpagospesos', 'Visa_Finiciomora',
        'cdescubierto_preacordado', 'rankn_ctarjeta_visa_transacciones', 'mpasivos_margen_max6',
        'mcuenta_corriente', 'Visa_msaldopesos', 'rankp_cdescubierto_preacordado',
        'mprestamos_personales', 'rankp_cpayroll_trx', 'ctarjeta_master_ratioavg6',
        'ccomisiones_mantenimiento_tend6', 'mcaja_ahorro_max6', 'cdescubierto_preacordado_min6',
        'rankp_mprestamos_personales', 'rankn_mpayroll', 'cproductos_tend6', 'mpayroll_sobre_edad_lag1',
        'mautoservicio', 'rankn_mprestamos_personales', 'mcomisiones_mantenimiento_ratioavg6',
        'rankp_Visa_mconsumospesos', 'ctarjeta_visa_delta2', 'mpayroll_ratioavg6',
        'mcaja_ahorro_ratioavg6', 'Master_status', 'cpayroll_trx_ratioavg6', 'Visa_mconsumospesos',
        'Visa_delinquency', 'mprestamos_personales_ratioavg6', 'foto_mes', 'mprestamos_personales_min6',
        'mcuenta_corriente_tend6', 'mcuentas_saldo_lag1', 'rankp_Visa_mpagominimo', 'ctarjeta_visa',
        'ctarjeta_debito_transacciones', 'mcuentas_saldo_tend6', 'cprestamos_personales_ratioavg6',
        'rankp_ccaja_ahorro', 'mcuenta_corriente_ratioavg6', 'rankp_cprestamos_personales',
        'cproductos_delta1', 'mpayroll_lag1', 'Visa_msaldototal', 'mpayroll_lag2', 'rankp_cproductos',
        'rankn_cpayroll_trx', 'ctrx_quarter_lag2', 'rankp_mpayroll', 'cproductos_delta2',
        'mpayroll_sobre_edad_ratioavg6', 'mrentabilidad_annual_max6', 'rankp_tcallcenter',
        'ccomisiones_mantenimiento_ratioavg6', 'Visa_mpagospesos_ratioavg6', 'ctarjeta_master',
        'rankp_mextraccion_autoservicio', 'rankp_cproductos_1', 'cdescubierto_preacordado_delta2',
        'mautoservicio_ratioavg6', 'rankp_Visa_cconsumos', 'Visa_msaldototal_lag1',
        'rankp_mactivos_margen', 'Visa_status', 'mcuenta_corriente_delta2', 'mpasivos_margen',
        'rankn_Visa_msaldototal', 'mcaja_ahorro_tend6', 'ctrx_quarter_normalizado_ratioavg6',
        'mpasivos_margen_tend6', 'mrentabilidad_annual', 'cpayroll_trx_lag2', 'rankp_mautoservicio',
        'mprestamos_personales_lag2', 'rankn_cproductos_1', 'Visa_Finiciomora_tend6',
        'rankp_mtarjeta_master_consumo', 'rankp_Master_mconsumospesos', 'cproductos',
        'Visa_delinquency_tend6', 'rankn_cdescubierto_preacordado', 'rankp_Visa_mpagospesos',
        'rankn_cprestamos_personales', 'mcuentas_saldo_lag2', 'mtarjeta_visa_consumo_tend6',
        'rankn_mautoservicio', 'mprestamos_personales_delta1', 'rankn_mactivos_margen',
        'mactivos_margen', 'rankp_Master_msaldopesos', 'cpayroll_trx_lag1', 'mprestamos_personales_max6',
        'rankp_ccallcenter_transacciones', 'Visa_mpagominimo', 'ctarjeta_debito_transacciones_ratioavg6',
        'ctarjeta_visa_transacciones_tend6', 'internet', 'mcuentas_saldo_ratioavg6',
        'rankp_ccaja_seguridad_1', 'ctrx_quarter_normalizado_delta1', 'mextraccion_autoservicio',
        'rankp_mcomisiones_mantenimiento_1', 'mcomisiones_mantenimiento_delta2',
        'ccomisiones_mantenimiento', 'ctrx_quarter_tend6', 'mprestamos_personales_lag1',
        'Master_delinquency', 'mcomisiones_mantenimiento_tend6', 'mcuenta_corriente_min6',
        'mpayroll_sobre_edad_lag2', 'cdescubierto_preacordado_tend6', 'rankn_Visa_mpagospesos',
        'Visa_status_tend6', 'rankp_mcuenta_debitos_automaticos', 'rankp_ccajas_consultas',
        'rankp_Master_msaldototal', 'Master_status_tend6', 'rankn_Visa_msaldopesos',
        'mrentabilidad_annual_lag2', 'rankp_ccomisiones_mantenimiento', 'mtarjeta_master_consumo',
        'rankp_Visa_msaldototal', 'mpayroll_tend6', 'Visa_delinquency_ratioavg6',
        'cprestamos_personales_min6', 'rankp_Visa_status', 'mactivos_margen_max6',
        'rankp_mplazo_fijo_dolares', 'ctarjeta_master_lag1', 'rankn_ccomisiones_otras',
        'rankp_mcuenta_debitos_automaticos_1', 'ctrx_quarter_delta1', 'rankp_ccuenta_debitos_automaticos',
        'Visa_msaldopesos_ratioavg6', 'ctarjeta_master_delta2', 'mpasivos_margen_delta1',
        'ccomisiones_mantenimiento_delta2', 'rankn_Visa_mpagominimo', 'Visa_mpagospesos_lag1',
        'mcuenta_corriente_lag1', 'rankp_Master_cconsumos', 'mcuentas_saldo_delta2',
        'cprestamos_personales', 'Visa_msaldototal_ratioavg6', 'Visa_delinquency_lag1',
        'rankn_ccaja_ahorro', 'rankp_mpagomiscuentas', 'mpayroll_sobre_edad_tend6',
        'mcuentas_saldo_delta1', 'mtransferencias_recibidas', 'Visa_delinquency_delta2',
        'rankp_ccaja_seguridad', 'Visa_Finiciomora_lag1', 'mrentabilidad_max6',
        'rankn_mrentabilidad_annual', 'rankn_mpagomiscuentas', 'mactivos_margen_min6',
        'rankp_cpagomiscuentas', 'mpagomiscuentas', 'rankp_mrentabilidad_annual', 'mcomisiones_lag2',
        'Visa_cconsumos', 'internet_lag2', 'rankp_Master_mpagominimo', 'cprestamos_personales_tend6',
        'mpayroll_sobre_edad_delta2', 'ctarjeta_visa_transacciones_max6',
        'cextraccion_autoservicio_ratioavg6', 'Visa_mpagominimo_tend6', 'mpayroll_delta2',
        'rankp_mtransferencias_emitidas', 'ctarjeta_visa_debitos_automaticos_ratioavg6',
        'rankp_Master_mfinanciacion_limite_1', 'mpasivos_margen_ratioavg6', 'rankp_Master_mpagospesos',
        'cprestamos_personales_lag2', 'rankp_mtransferencias_recibidas', 'ctarjeta_visa_tend6',
        'internet_lag1', 'cpagomiscuentas', 'ctrx_quarter_normalizado_lag2', 'rankn_Visa_status',
        'Visa_mpagominimo_delta1', 'mcomisiones_mantenimiento', 'Visa_cconsumos_ratioavg6',
        'rankp_cextraccion_autoservicio', 'rankn_cproductos', 'mcomisiones_otras_ratioavg6',
        'rankp_mtransferencias_recibidas_1', 'mtarjeta_master_consumo_ratioavg6', 'internet_ratioavg6',
        'ccomisiones_otras_ratioavg6', 'Master_msaldopesos', 'Master_mfinanciacion_limite',
        'ctrx_quarter_delta2', 'Master_mpagospesos_ratioavg6', 'mpasivos_margen_lag2',
        'ctrx_quarter_normalizado_delta2', 'cliente_edad_lag2', 'rankp_mrentabilidad',
        'mrentabilidad_annual_lag1', 'Visa_fultimo_cierre_tend6', 'Visa_fechaalta',
        'mactivos_margen_tend6', 'mcaja_ahorro_delta2', 'mactivos_margen_ratioavg6',
        'rankp_mcomisiones_mantenimiento', 'Visa_mpagosdolares', 'mcuentas_saldo_min6',
        'Master_mpagominimo', 'Visa_msaldototal_lag2', 'mactivos_margen_lag1', 'ccaja_ahorro',
        'internet_max6', 'rankp_Visa_mconsumosdolares', 'mcuentas_saldo_max6',
        'mcomisiones_mantenimiento_delta1', 'mttarjeta_visa_debitos_automaticos_ratioavg6',
        'mpasivos_margen_lag1', 'Visa_mpagominimo_lag2', 'Visa_mpagominimo_delta2',
        'mcuenta_corriente_max6', 'cpayroll_trx_tend6', 'mcomisiones_mantenimiento_lag2',
        'Visa_msaldopesos_lag1', 'mcomisiones_ratioavg6', 'Visa_fechaalta_lag1', 'Master_fechaalta',
        'cextraccion_autoservicio', 'rankn_mcomisiones_mantenimiento_1', 'Master_Fvencimiento',
        'ccaja_ahorro_max6', 'cprestamos_personales_delta2', 'Visa_mpagospesos_tend6',
        'Master_status_delta2', 'ctarjeta_master_transacciones_ratioavg6',
        'mprestamos_personales_tend6', 'mcuenta_corriente_delta1', 'cliente_antiguedad',
        'cliente_antiguedad_lag1', 'Master_mpagominimo_tend6', 'Visa_mfinanciacion_limite',
        'mrentabilidad', 'rankp_Master_mlimitecompra_1', 'cpagomiscuentas_ratioavg6',
        'rankp_thomebanking', 'ccuenta_debitos_automaticos_ratioavg6', 'ccomisiones_otras',
        'rankp_ccomisiones_otras', 'rankn_mtransferencias_recibidas', 'mtransferencias_recibidas_lag1',
        'Visa_msaldopesos_lag2', 'cliente_edad_min6', 'Visa_mconsumospesos_lag2',
        'rankn_mtarjeta_master_consumo', 'cprestamos_personales_lag1', 'mrentabilidad_lag1',
        'Master_Fvencimiento_ratioavg6', 'rankn_mcuenta_debitos_automaticos_1',
        'ccaja_seguridad_lag2', 'mtransferencias_recibidas_ratioavg6',
        'ctarjeta_master_transacciones', 'mtarjeta_visa_consumo_delta2', 'Master_msaldototal',
        'rankp_Visa_mfinanciacion_limite_1', 'mcomisiones_max6', 'mrentabilidad_annual_delta2',
        'ccomisiones_mantenimiento_lag2', 'mpayroll_sobre_edad_delta1',
        'ctarjeta_visa_transacciones_delta1', 'rankp_ccajas_transacciones',
        'rankp_mttarjeta_visa_debitos_automaticos', 'ctarjeta_debito_transacciones_max6',
        'mcomisiones_otras_lag2', 'Master_mpagominimo_lag1', 'Master_fechaalta_ratioavg6',
        'Visa_mpagominimo_lag1', 'rankp_Master_mfinanciacion_limite', 'Master_fultimo_cierre',
        'cdescubierto_preacordado_delta1', 'mautoservicio_max6', 'cproductos_min6',
        'rankp_ccajas_otras', 'Master_mpagominimo_delta2', 'mttarjeta_visa_debitos_automaticos_tend6',
        'mcomisiones_min6', 'mcomisiones_mantenimiento_lag1', 'rankp_thomebanking_1',
        'mactivos_margen_lag2', 'Master_Fvencimiento_lag1', 'Visa_fechaalta_lag2',
        'rankp_Visa_mfinanciacion_limite', 'ccomisiones_mantenimiento_delta1',
        'rankp_Master_mlimitecompra', 'mcomisiones_otras', 'cproductos_lag2', 'cmobile_app_trx',
        'cdescubierto_preacordado_lag1', 'cproductos_lag1', 'rankn_mcuenta_debitos_automaticos',
        'mpagomiscuentas_ratioavg6', 'matm_ratioavg6', 'Visa_mlimitecompra', 'Master_msaldototal_tend6',
        'active_quarter_min6', 'Master_msaldopesos_lag2', 'Visa_madelantodolares',
        'mtarjeta_visa_consumo_delta1', 'cmobile_app_trx_lag2', 'tcallcenter_ratioavg6',
        'mrentabilidad_annual_min6', 'Visa_cadelantosefectivo', 'mrentabilidad_lag2', 'mcomisiones',
        'mpagomiscuentas_lag1', 'ctarjeta_master_transacciones_max6', 'rankp_tmobile_app',
        'mpayroll_delta1', 'mcuenta_debitos_automaticos_ratioavg6', 'ctarjeta_visa_transacciones_min6',
        'Master_mconsumospesos', 'ccaja_ahorro_lag1', 'ctransferencias_recibidas_ratioavg6',
        'Master_mconsumototal', 'Visa_Fvencimiento_tend6', 'rankn_mcomisiones_mantenimiento',
        'cprestamos_personales_max6', 'kmes_tend6', 'chomebanking_transacciones_ratioavg6',
        'Visa_mpagominimo_ratioavg6', 'mcuenta_corriente_lag2', 'ctarjeta_visa_transacciones_delta2',
        'Visa_fechaalta_ratioavg6', 'rankp_mcaja_ahorro_dolares', 'Visa_Fvencimiento_lag2',
        'rankp_Visa_Fvencimiento', 'Master_fechaalta_lag2', 'Master_mpagospesos',
        'ccallcenter_transacciones_ratioavg6', 'Master_delinquency_lag1', 'rankn_Master_mpagominimo',
        'cliente_edad_max6', 'ctrx_quarter_normalizado_tend6', 'Visa_mpagado', 'cmobile_app_trx_delta2',
        'mtarjeta_master_consumo_tend6', 'mcomisiones_lag1', 'Master_Fvencimiento_tend6',
        'ctransferencias_recibidas_lag1', 'Visa_mfinanciacion_limite_lag1',
        'ccomisiones_otras_tend6', 'mtransferencias_recibidas_tend6', 'Master_mpagosdolares',
        'kmes_lag1', 'rankp_ctarjeta_visa_debitos_automaticos',
        'ctarjeta_visa_debitos_automaticos_tend6', 'mpagomiscuentas_tend6', 'cliente_antiguedad_min6',
        'rankp_mcomisiones_otras', 'cpayroll_trx_delta1', 'Visa_fultimo_cierre_lag2',
        'cliente_antiguedad_lag2', 'Master_mpagominimo_lag2', 'Visa_mfinanciacion_limite_lag2',
        'Master_fechaalta_lag1', 'rankn_Master_msaldototal', 'internet_min6',
        'cliente_antiguedad_ratioavg6', 'Visa_msaldototal_tend6', 'mcaja_ahorro_dolares_delta1',
        'Master_msaldototal_ratioavg6', 'rankp_chomebanking_transacciones', 'Visa_msaldototal_delta2',
        'mcomisiones_tend6', 'Master_cconsumos_delta2', 'chomebanking_transacciones_lag2',
        'cliente_antiguedad_max6', 'Visa_msaldototal_delta1', 'Visa_fultimo_cierre',
        'Master_mpagominimo_delta1', 'cdescubierto_preacordado_lag2',
        'rankn_chomebanking_transacciones', 'ccaja_ahorro_min6', 'ccomisiones_otras_lag1',
        'Master_madelantodolares', 'mcaja_ahorro_dolares_delta2', 'mcuenta_debitos_automaticos_lag2',
        'mcaja_ahorro_dolares_tend6', 'Master_mconsumospesos_lag1', 'ccuenta_debitos_automaticos',
        'Visa_mconsumospesos_tend6', 'mextraccion_autoservicio_ratioavg6', 'thomebanking_ratioavg6',
        'rankn_Visa_mfinanciacion_limite', 'ccomisiones_otras_lag2', 'Master_fechaalta_tend6',
        'ccaja_ahorro_ratioavg6', 'kmes_ratioavg6', 'Visa_mfinanciacion_limite_tend6',
        'mautoservicio_min6', 'Visa_Fvencimiento', 'cliente_edad_lag1', 'Visa_msaldopesos_delta2',
        'ctarjeta_debito_transacciones_lag2', 'Master_mlimitecompra', 'mtarjeta_master_consumo_max6',
        'mtarjeta_visa_consumo_lag2', 'Master_cconsumos', 'mcuenta_debitos_automaticos',
        'ccaja_ahorro_lag2', 'ctransferencias_recibidas', 'Visa_fechaalta_tend6',
        'mpasivos_margen_min6', 'Visa_mlimitecompra_lag1', 'ccajas_transacciones',
        'mprestamos_personales_delta2', 'Master_msaldopesos_delta2', 'mactivos_margen_delta2',
        'cmobile_app_trx_lag1', 'mrentabilidad_delta1', 'Visa_mfinanciacion_limite_ratioavg6',
        'Master_mconsumospesos_tend6', 'Visa_status_delta1', 'ccajas_consultas',
        'mtarjeta_visa_consumo_max6', 'internet_tend6', 'mrentabilidad_min6', 'kmes',
        'ccomisiones_mantenimiento_lag1', 'Master_mfinanciacion_limite_lag2',
        'ccajas_transacciones_ratioavg6', 'Visa_mconsumosdolares_ratioavg6',
        'Visa_mconsumospesos_lag1', 'Visa_msaldopesos_tend6', 'mcuenta_debitos_automaticos_tend6',
        'Master_msaldopesos_ratioavg6', 'Master_cadelantosefectivo', 'Master_mconsumosdolares',
        'Visa_madelantopesos', 'matm', 'cliente_edad', 'mcaja_ahorro_dolares_min6',
        'Visa_mconsumosdolares', 'Visa_mlimitecompra_lag2', 'ccajas_consultas_ratioavg6',
        'mcaja_ahorro_dolares_ratioavg6', 'Master_msaldototal_delta1', 'catm_trx_ratioavg6',
        'Visa_mfinanciacion_limite_delta1', 'Visa_mlimitecompra_tend6', 'Master_msaldototal_delta2',
        'Master_mfinanciacion_limite_delta2', 'mautoservicio_delta2', 'Visa_mconsumospesos_delta2',
        'Master_mlimitecompra_tend6', 'cmobile_app_trx_ratioavg6', 'tmobile_app',
        'Master_mlimitecompra_lag1', 'rankn_mplazo_fijo_dolares', 'mcaja_ahorro_dolares',
        'mcaja_ahorro_dolares_max6', 'tmobile_app_delta1', 'ctarjeta_master_transacciones_lag1',
        'mrentabilidad_tend6', 'cpagomiscuentas_tend6', 'cmobile_app_trx_delta1',
        'Master_mfinanciacion_limite_tend6', 'rankp_Visa_mlimitecompra', 'tmobile_app_delta2',
        'catm_trx', 'ctarjeta_master_debitos_automaticos', 'ctransferencias_emitidas',
        'ccaja_seguridad', 'mtransferencias_emitidas', 'ctarjeta_visa_debitos_automaticos',
        'Visa_mfinanciacion_limite_delta2', 'Visa_cconsumos_delta2', 'kmes_lag2',
        'chomebanking_transacciones', 'ctarjeta_debito', 'cmobile_app_trx_tend6',
        'Master_Fvencimiento_lag2', 'cforex', 'matm_tend6', 'Visa_mconsumosdolares_tend6',
        'ccajas_otras_tend6', 'mcheques_depositados_rechazados_tend6', 'cpagodeservicios_tend6',
        'ccheques_depositados_rechazados_tend6', 'mcajeros_propios_descuentos_tend6',
        'mtarjeta_visa_descuentos_tend6', 'Master_cconsumos_tend6', 'ccheques_emitidos_tend6',
        'cforex_buy_tend6', 'ccajas_consultas_tend6', 'Master_msaldopesos_tend6',
        'ccheques_emitidos_rechazados_tend6', 'mforex_sell_lag1', 'mcomisiones_otras_lag1',
        'mprestamos_hipotecarios_lag1', 'cseguro_vida_lag2', 'ctarjeta_visa_lag2',
        'Visa_cconsumos_lag2', 'Visa_cconsumos_lag1', 'chomebanking_transacciones_lag1',
        'ctransferencias_emitidas_delta2', 'Visa_Fvencimiento_lag1',
        'Master_mfinanciacion_limite_lag1', 'Visa_msaldodolares_lag1',
        'mtarjeta_master_consumo_delta2', 'mplazo_fijo_dolares_delta2', 'cforex_buy',
        'ctarjeta_visa_debitos_automaticos_delta2', 'tcallcenter_tend6',
        'ccallcenter_transacciones_tend6', 'mcheques_emitidos_rechazados_tend6',
        'cplazo_fijo_tend6', 'ccheques_depositados_tend6', 'ccaja_seguridad_lag1',
        'mcheques_emitidos_lag2', 'thomebanking', 'mprestamos_prendarios_lag1',
        'mtransferencias_emitidas_lag1', 'matm_other_lag1',
        'ctarjeta_visa_debitos_automaticos_lag1', 'Visa_mpagado_lag1',
        'ctarjeta_master_transacciones_tend6', 'ctarjeta_master_debitos_automaticos_tend6',
        'mtarjeta_master_descuentos', 'Visa_Finiciomora_ratioavg6', 'tmobile_app_ratioavg6',
        'mplazo_fijo_dolares_ratioavg6', 'mcheques_emitidos', 'ccaja_seguridad_ratioavg6',
        'ccheques_depositados_rechazados_ratioavg6', 'mprestamos_prendarios_ratioavg6',
        'ccheques_emitidos_rechazados_ratioavg6', 'Visa_status_ratioavg6',
        'ctarjeta_debito_ratioavg6', 'ctarjeta_master_debitos_automaticos_ratioavg6',
        'mttarjeta_master_debitos_automaticos_ratioavg6', 'mtarjeta_visa_descuentos_ratioavg6',
        'Master_mpagosdolares_ratioavg6', 'ctarjeta_visa_descuentos_ratioavg6', 'cliente_vip',
        'Master_msaldodolares_ratioavg6', 'active_quarter', 'mtarjeta_visa_descuentos', 'mforex_buy',
        'cforex_sell', 'mforex_sell', 'ccheques_depositados', 'mcheques_depositados',
        'ccheques_emitidos', 'ctarjeta_debito_tend6', 'ccheques_depositados_rechazados',
        'mcheques_depositados_rechazados', 'ccheques_emitidos_rechazados',
        'mcheques_emitidos_rechazados', 'tcallcenter', 'ccallcenter_transacciones',
        'ccajas_extracciones_tend6', 'ctransferencias_recibidas_lag2', 'ccajas_extracciones',
        'ccajas_otras', 'chomebanking_transacciones_tend6', 'thomebanking_tend6',
        'minversion2_tend6', 'mforex_buy_tend6', 'mtransferencias_emitidas_tend6',
        'cprestamos_hipotecarios', 'Visa_mconsumosdolares_lag2',
        'mttarjeta_visa_debitos_automaticos_lag2', 'tcuentas_lag2', 'mprestamos_prendarios_lag2',
        'Master_fultimo_cierre_lag2', 'catm_trx_other_lag2', 'ccajas_depositos',
        'mprestamos_prendarios', 'numero_de_cliente', 'mprestamos_hipotecarios', 'cplazo_fijo',
        'ctransferencias_emitidas_ratioavg6', 'mcheques_emitidos_rechazados_ratioavg6',
        'mtransferencias_emitidas_ratioavg6', 'Master_Finiciomora_ratioavg6',
        'Master_delinquency_ratioavg6', 'mcheques_depositados_rechazados_ratioavg6',
        'Master_mconsumosdolares_ratioavg6', 'ccheques_emitidos_ratioavg6',
        'ccajas_otras_ratioavg6', 'catm_trx_other', 'cplazo_fijo_ratioavg6',
        'mcheques_emitidos_ratioavg6', 'active_quarter_ratioavg6',
        'ccheques_depositados_ratioavg6', 'mforex_buy_ratioavg6',
        'ccajeros_propios_descuentos_ratioavg6', 'cforex_ratioavg6',
        'ccajas_extracciones_ratioavg6', 'rankn_chomebanking_transacciones_1',
        'rankn_tcallcenter', 'rankn_ccallcenter_transacciones', 'rankn_mtransferencias_emitidas',
        'rankn_ctarjeta_visa_debitos_automaticos', 'rankn_Master_mlimitecompra',
        'mcheques_depositados_ratioavg6', 'ctarjeta_visa_debitos_automaticos_lag2',
        'cinversion2', 'Master_mlimitecompra_lag2', 'mrentabilidad_annual_tend6',
        'mplazo_fijo_dolares', 'mplazo_fijo_pesos', 'cinversion1', 'minversion1_pesos',
        'minversion1_dolares', 'cprestamos_prendarios', 'minversion2', 'cseguro_vida',
        'cseguro_auto', 'mcomisiones_otras_tend6', 'cseguro_accidentes_personales', 'mpayroll2',
        'cpayroll2_trx', 'mttarjeta_visa_debitos_automaticos', 'mttarjeta_master_debitos_automaticos',
        'cpagodeservicios', 'mpagodeservicios', 'ccajeros_propios_descuentos',
        'mcajeros_propios_descuentos', 'ctarjeta_visa_descuentos', 'tcuentas', 'ccuenta_corriente',
        'mcuenta_corriente_adicional', 'mcaja_ahorro_adicional', 'cseguro_vivienda',
        'Visa_mlimitecompra_delta1', 'ctarjeta_visa_delta1', 'mplazo_fijo_dolares_delta1',
        'ctransferencias_emitidas_delta1', 'thomebanking_delta1', 'mrentabilidad_annual_delta1',
        'mcomisiones_otras_delta1', 'Visa_msaldodolares', 'Visa_Finiciomora_delta1',
        'ccaja_ahorro_delta1', 'Master_Finiciomora_delta1', 'cprestamos_personales_delta1',
        'mprestamos_hipotecarios_max6', 'rankn_ccajas_consultas', 'mprestamos_prendarios_max6',
        'cseguro_vida_max6', 'ccaja_ahorro_tend6', 'ctarjeta_master_descuentos', 'matm_other',
        'Master_Finiciomora', 'Master_msaldodolares', 'Master_madelantopesos', 'Master_mpagado',
        'Master_delinquency_tend6', 'ccuenta_debitos_automaticos_tend6', 'Master_Finiciomora_tend6',
        'mplazo_fijo_dolares_tend6', 'tmobile_app_tend6', 'tcuentas_max6',
        'rankn_ctransferencias_emitidas', 'tcuentas_min6', 'mprestamos_prendarios_min6',
        'ctarjeta_debito_min6', 'mprestamos_hipotecarios_min6', 'cseguro_vida_min6',
        'ccallcenter_transacciones_delta1', 'Visa_cconsumos_delta1', 'Visa_msaldodolares_delta1',
        'mcheques_emitidos_rechazados_delta1', 'ccuenta_debitos_automaticos_delta1',
        'mprestamos_prendarios_delta2', 'tcuentas_lag1', 'chomebanking_transacciones_delta1',
        'rankp_ctarjeta_debito', 'rankp_chomebanking_transacciones_1', 'rankp_Visa_mpagado',
        'ccallcenter_transacciones_delta2', 'ccaja_ahorro_delta2',
        'mcuenta_debitos_automaticos_delta2', 'ccuenta_debitos_automaticos_delta2',
        'Master_delinquency_delta2', 'tcallcenter_delta2', 'Visa_status_delta2',
        'mttarjeta_visa_debitos_automaticos_delta2', 'chomebanking_transacciones_delta2',
        'rankn_Visa_mlimitecompra', 'rankn_thomebanking', 'mcheques_emitidos_rechazados_lag1'
    ],
}

# Directorios
LOGS_DIR = "logs/compare_standalone"
LOCAL_DATASET_PATH = "dataset.parquet"  # Ruta local donde se guardará el dataset descargado

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _ensure_positive_odd(value: float) -> int:
    """Asegura que un valor sea positivo e impar."""
    n = max(1, int(round(value)))
    if n % 2 == 0:
        n += 1
    return n

DEFAULT_PLATEAU_WINDOW = _ensure_positive_odd(BASE_PLATEAU_WINDOW)
PRIVATE_PLATEAU_WINDOW = _ensure_positive_odd(BASE_PLATEAU_WINDOW * 0.7)
PUBLIC_PLATEAU_WINDOW = _ensure_positive_odd(BASE_PLATEAU_WINDOW * 0.3)

YELLOW = "\033[93m"
RESET = "\033[0m"

def get_expanded_seeds(n_seeds_needed):
    """
    Expande SEEDS_BASE si es necesario para obtener n_seeds_needed seeds.
    Si no hay suficientes seeds, genera nuevos sumando +1 al seed más grande.
    """
    expanded_seeds = list(SEEDS_BASE)
    
    if len(expanded_seeds) < n_seeds_needed:
        max_seed = max(expanded_seeds) if expanded_seeds else 0
        seeds_needed = n_seeds_needed - len(expanded_seeds)
        for i in range(1, seeds_needed + 1):
            expanded_seeds.append(max_seed + i)
    
    return expanded_seeds[:n_seeds_needed]

def download_dataset_from_gcp(gcp_url: str, local_path: str):
    """
    Descarga el dataset desde un bucket de GCP.
    
    Args:
        gcp_url: URL del bucket (ej: gs://bucket-name/path/to/file.parquet)
        local_path: Ruta local donde guardar el archivo
    """
    logger.info(f"Descargando dataset desde {gcp_url} a {local_path}")
    
    # Remover gs://
    path_parts = gcp_url[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.download_to_filename(local_path)
    logger.info(f"Dataset descargado exitosamente a {local_path}")

def load_dataset(path_parquet: str, months: list):
    """
    Devuelve el dataset como un df de Polars.
    - Crea columna `y_train` juntando BAJA+1 y BAJA+2.
    - Crea columna `y_true` juntando BAJA+1 y CONTINUA.
    - Crea columna `w_train` dando un peso distinto a cada clase.
    """
    if isinstance(months, str):
        months = [months]
    
    df_lazy = (
        pl.scan_parquet(path_parquet, low_memory=True)
        .filter(pl.col("foto_mes").is_in(months))
        .with_columns([
            # Binary label
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0)
            .otherwise(1)
            .alias("y_train"),
            
            # Binary label
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
            .otherwise(0)
            .alias("y_true"),
            
            # Weights
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(1)
            .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
            .when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
            .otherwise(None)
            .alias("w_train")
        ])
    )
    
    df = df_lazy.collect()
    return df

def load_dataset_undersampling_efficient(
    path_parquet: str,
    months: list,
    fraction: float = 0.1,
    seed: int = 480,
    stratified: bool = False
) -> pl.DataFrame:
    """
    Carga el dataset con undersampling. Quedó una key stratified que no conviene usar porque rompe todo
    """
    if isinstance(months, str):
        months = [months]
    
    logger.info(f"Cargando dataset desde {path_parquet}")
    
    df_lazy = (
        pl.scan_parquet(path_parquet, low_memory=True)
        .filter(pl.col("foto_mes").is_in(months))
        .with_columns([
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0).otherwise(1).alias("y_train"),
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1).otherwise(0).alias("y_true"),
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(1)
             .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
             .when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
             .otherwise(None)
             .alias("w_train")
        ])
    )
    
    logger.info("Aplicando undersamppling...")
    
    if stratified:
        df_lazy = (
            df_lazy
            .with_columns(
                pl.when(pl.col("clase_ternaria") == "CONTINUA")
                .then(
                    ((pl.col("numero_de_cliente").hash() + pl.lit(seed) + 
                      pl.col("foto_mes").hash()).hash() % 1000000) / 1000000.0
                )
                .otherwise(None)
                .alias("_hash_val")
            )
            .filter(
                (pl.col("clase_ternaria") != "CONTINUA")
                | (pl.col("_hash_val") <= fraction)
            )
            .select(pl.all().exclude(["_hash_val"]))
        )
    else:
        df_lazy = (
            df_lazy
            .with_columns(
                pl.when(pl.col("clase_ternaria") == "CONTINUA")
                .then(
                    ((pl.col("numero_de_cliente").hash() + pl.lit(seed)).hash() % 1000000) / 1000000.0
                )
                .otherwise(None)
                .alias("_hash_val")
            )
            .filter(
                (pl.col("clase_ternaria") != "CONTINUA")
                | (pl.col("_hash_val") <= fraction)
            )
            .select(pl.all().exclude(["_hash_val"]))
        )
    
    df = df_lazy.collect()
    
    logger.info("Calculando conteos post-undersampling por mes y clase...")
    counts_df = df.group_by("foto_mes", "clase_ternaria").agg(
        pl.count().alias("registros")
    ).sort("foto_mes", "clase_ternaria")
    logger.info(f"--- Conteos finales Post-Undersampling ---\n{counts_df}\n--------------------------------------------")
    logger.info(f"Columnas del dataset muestreado: {df.width}")
    logger.info(f"Filas del dataset muestreado: {df.height}")
    
    return df

def train_model(params: dict, dtrain, features: list[str]):
    """
    Entrena el modelo final con los mejores hiperparámetros usando LightGBM.
    """
    params['deterministic'] = True
    params['bagging_fraction_seed'] = params['seed']
    params['feature_fraction_seed'] = params['seed']
    
    modelo = lgb.train(params, dtrain)
    logger.info("Entrenamiento completado")
    return modelo

def predict_testset(modelo: lgb.Booster, months: list, df: pl.DataFrame):
    """
    Genera predicciones
    """
    df = df.filter(pl.col("foto_mes").is_in(months))
    clientes = df["numero_de_cliente"].to_numpy()
    X = df.select(modelo.feature_name()).to_numpy()
    y_true = df.select('y_true')
    
    logger.info(f"Total de filas a predecir: {df.height:,}")
    logger.info(f"Generando predicciones. Mes: {months}")
    y_pred = modelo.predict(X)
    
    resultados = pl.DataFrame({
        "numero_de_cliente": clientes,
        "y_pred": y_pred,
        "y_true": y_true
    })
    
    return resultados

def merge_predictions(pred_acumuladas: pl.DataFrame, n_submissions=10000):
    """Promedia predicciones de múltiples modelos, se usa esta fc para el semillerío"""
    cols_pred = [c for c in pred_acumuladas.columns if c.startswith('y_pred_')]
    pred_final = pred_acumuladas.with_columns(
        (pl.sum_horizontal(cols_pred) / len(cols_pred)).alias('y_pred_mean')
    )
    
    pred_final = pred_final.with_columns(
        pl.Series('rank', pred_final['y_pred_mean'].rank(descending=True))
    )
    pred_final = pred_final.with_columns(
        pl.when(pl.col('rank') <= n_submissions)
        .then(1)
        .otherwise(0)
        .alias('predict')
    )
    
    return pred_final

def get_cum_gan(y_pred, y_true):
    """Dado un vector de predicciones y otro de true labels, devuelve el vector de ganancia acumulada"""
    ganancia = np.where(y_true == 1, GANANCIA_ACIERTO, 0) - np.where(y_true == 0, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    return np.cumsum(ganancia)

def get_max_plateau_gain(cum_gan_vector: np.ndarray, window: int = 2001) -> tuple[float, int]:
    """
    Devuelve la ganancia máxima en meseta y el índice asociado usando una media móvil
    centrada de tamaño impar 'window' sobre la curva de ganancia acumulada ya ordenada. Esta es una versión de la fc usada por Gustavo
    """
    arr = np.asarray(cum_gan_vector, dtype=float)
    
    if arr.ndim != 1:
        raise ValueError("cum_gan_vector must be a one-dimensional array")
    
    n = arr.size
    if n == 0:
        return float("nan"), -1
    
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    
    if n < window:
        best_idx = int(np.argmax(arr))
        return float(arr[best_idx]), best_idx
    
    half_window = window // 2
    
    prefix = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    window_sums = prefix[window:] - prefix[:-window]
    rolling_mean = window_sums / window
    
    smoothed = np.full(n, np.nan, dtype=float)
    smoothed[half_window:n - half_window] = rolling_mean
    
    valid_mask = ~np.isnan(smoothed)
    if not np.any(valid_mask):
        best_idx = int(np.argmax(arr))
        return float(arr[best_idx]), best_idx
    
    best_idx = int(np.nanargmax(smoothed))
    max_plateau = float(smoothed[best_idx])
    
    return max_plateau, best_idx

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    tz_ba = ZoneInfo("America/Argentina/Buenos_Aires")
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now(tz_ba).strftime("%Y-%m-%d_%H-%M-%S")
    nombre_log = f"compare_standalone_{EXPERIMENT_NAME}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, nombre_log), mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Los logs se guardan en la carpeta: {LOGS_DIR}")
    
    # Descargar dataset si no existe localmente
    if not os.path.exists(LOCAL_DATASET_PATH):
        download_dataset_from_gcp(DATASET_GCP_URL, LOCAL_DATASET_PATH)
    else:
        logger.info(f"Dataset ya existe localmente en {LOCAL_DATASET_PATH}")
    
    # Obtener seeds expandidos
    experiment_seeds = get_expanded_seeds(N_EXPERIMENTS)
    results = {}
    feature_importances_all = {}
    
    # Detectar dinámicamente todos los modelos en la configuración
    model_names = list(MODELS_CONFIG.keys())
    model_names.sort()
    
    logger.info(f"Comparando {len(model_names)} modelos: {model_names}")
    
    # Cargar dataset de validación
    df_valid = load_dataset(path_parquet=LOCAL_DATASET_PATH, months=VAL_MONTH)
    
    # Entrenamiento y predicciones
    semillerio_por_modelo = {}
    df_train_cache = {}
    
    for model_name in model_names:
        scores = []
        subs = []
        preds = []
        cum_gans = []
        feature_importances = []
        
        model_config = MODELS_CONFIG[model_name]
        
        # Obtener semillerío
        semillerio = model_config.get("semillerio", 1)
        if semillerio is None:
            semillerio = 1
        semillerio_por_modelo[model_name] = semillerio
        
        # Preparar features
        features_all = []
        for features_name in model_config["chosen_features"]:
            if features_name in FEATURE_SUBSETS:
                features_all.extend(FEATURE_SUBSETS[features_name])
        features_all = list(set(features_all))
        
        # Preparar features para entrenamiento (sin clase_ternaria)
        features_train = features_all.copy()
        if 'clase_ternaria' in features_train:
            features_train.remove('clase_ternaria')
        
        # Obtener configuración de undersampling
        us_frac_raw = model_config.get("undersampling_fraction", None)
        if us_frac_raw is None or us_frac_raw == '' or us_frac_raw == 1.0:
            use_us = False
            us_frac = None
        elif 0 < us_frac_raw < 1:
            use_us = True
            us_frac = us_frac_raw
        else:
            use_us = False
            us_frac = None
            logger.warning(f"undersampling_fraction={us_frac_raw} no es válido para {model_name}. No se usará undersampling.")
        
        us_stratified = model_config.get("undersampling_stratified", False)
        
        MONTHS = model_config["months"]
        logger.info(f"=== MODELO: {model_name} ===")
        logger.info(f"MESES DE ENTRENAMIENTO: {MONTHS}")
        
        # Crear clave para el caché
        cache_key = (tuple(sorted(MONTHS)), use_us, us_frac, us_stratified)
        
        # Verificar si ya existe un df_train con la misma configuración en el caché
        if cache_key in df_train_cache:
            logger.info(f"✓ Reutilizando dataset del caché para {model_name}")
            df_train = df_train_cache[cache_key]
        else:
            if use_us:
                logger.info(f"Cargando dataset con undersampling (Fracción: {us_frac}, Estratificado: {us_stratified})")
                df_train = load_dataset_undersampling_efficient(
                    path_parquet=LOCAL_DATASET_PATH,
                    months=MONTHS,
                    fraction=us_frac,
                    seed=experiment_seeds[0],
                    stratified=us_stratified
                )
            else:
                logger.info(f"Cargando dataset sin undersampling")
                df_train = load_dataset(
                    path_parquet=LOCAL_DATASET_PATH,
                    months=MONTHS
                )
            
            df_train_cache[cache_key] = df_train
            logger.info(f"Dataset guardado en caché")
        
        # Crear dtrain con las features específicas de este modelo
        logger.info(f"Preparando dataset de entrenamiento con {len(features_train)} features")
        X_train = df_train.select(features_train).to_numpy()
        y_train = df_train["y_train"].to_numpy()
        w_train = df_train["w_train"].to_numpy()
        
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            feature_name=features_train,
            free_raw_data=True
        )
        
        del X_train, y_train, w_train
        gc.collect()
        
        for n_exp, seed in enumerate(experiment_seeds):
            logger.info(f"Entrenando {model_name} experimento {n_exp + 1}/{N_EXPERIMENTS} con seed base {seed}, semillerio={semillerio}")
            
            params = FIXED_PARAMS.copy()
            params.update(model_config["params"])
            n_submissions = model_config.get("n_submissions", None)
            
            if semillerio > 1:
                logger.info(f"Entrenando semillerio de {semillerio} modelos para {model_name} (Seed base {seed})")
                
                semillerio_seeds = [n_exp * 1000 + i for i in range(semillerio)]
                
                pred_acumuladas = None
                exp_feature_importances = []
                
                for sem_idx, sem_seed in enumerate(semillerio_seeds):
                    logger.info(f"  Entrenando modelo {sem_idx + 1}/{semillerio} del semillerio {model_name} con seed {sem_seed}")
                    params_sem = params.copy()
                    params_sem["seed"] = sem_seed
                    params_sem["verbose"] = -1
                    
                    model = train_model(
                        params=params_sem,
                        dtrain=dtrain,
                        features=features_train
                    )
                    
                    exp_feature_importances.append(
                        pd.Series(
                            model.feature_importance(importance_type="gain"),
                            index=model.feature_name(),
                            name=f"sem_seed_{sem_seed}"
                        )
                    )
                    
                    logger.info(f"Generando predicciones")
                    resultados = predict_testset(
                        modelo=model,
                        months=VAL_MONTH,
                        df=df_valid
                    )
                    
                    pred_df = resultados.select(['numero_de_cliente', 'y_pred']).clone()
                    pred_df = pred_df.rename({'y_pred': f'y_pred_{sem_seed}'})
                    
                    if pred_acumuladas is None:
                        pred_acumuladas = resultados.select(['numero_de_cliente', 'y_true']).clone()
                        pred_acumuladas = pred_acumuladas.join(pred_df, on='numero_de_cliente', how='inner')
                    else:
                        pred_acumuladas = pred_acumuladas.join(pred_df, on='numero_de_cliente', how='inner')
                    
                    del model
                    gc.collect()
                
                if n_submissions is None:
                    n_submissions = 10000
                pred_final = merge_predictions(pred_acumuladas, n_submissions=n_submissions)
                
                y_true_values = pred_final['y_true'].to_numpy() if hasattr(pred_final['y_true'], 'to_numpy') else np.array(pred_final['y_true'])
                y_pred_values = pred_final['y_pred_mean'].to_numpy()
                cum_gan = get_cum_gan(y_pred_values, y_true_values)
                cum_gans.append(cum_gan)
                
                if model_config.get("n_submissions", None):
                    subs_i = n_submissions
                    score = cum_gan[subs_i]
                else:
                    gan_plateau, subs_idx = get_max_plateau_gain(cum_gan, window=DEFAULT_PLATEAU_WINDOW)
                    score = float(gan_plateau)
                    subs_i = int(subs_idx)
                
                scores.append(score)
                subs.append(subs_i)
                preds.append(y_pred_values)
                
                fi_df = pd.concat(exp_feature_importances, axis=1)
                fi_mean = fi_df.mean(axis=1)
                fi = pd.Series(fi_mean, index=fi_df.index, name=f"seed_{seed}")
                feature_importances.append(fi)
                
                logger.info(f"Score {model_name} (Seed base {seed}, semillerio={semillerio}): {score} con {subs_i} envíos")
                
                del pred_acumuladas, pred_final
                gc.collect()
            else:
                params["seed"] = seed
                
                if use_us:
                    logger.info(f"Entrenando {model_name} (Seed {seed}) CON undersampling (Fracción: {us_frac})")
                else:
                    logger.info(f"Entrenando {model_name} (Seed {seed}) SIN undersampling")
                
                model = train_model(
                    params=params,
                    dtrain=dtrain,
                    features=features_train
                )
                
                fi = pd.Series(
                    model.feature_importance(importance_type="gain"),
                    index=model.feature_name(),
                    name=f"seed_{seed}"
                )
                feature_importances.append(fi)
                
                logger.info("Generando predicciones")
                resultados = predict_testset(
                    modelo=model,
                    months=VAL_MONTH,
                    df=df_valid
                )
                
                cum_gan = get_cum_gan(resultados["y_pred"], resultados["y_true"])
                cum_gans.append(cum_gan)
                
                if model_config.get("n_submissions", None):
                    subs_i = model_config["n_submissions"]
                    score = cum_gan[subs_i]
                else:
                    gan_plateau, subs_idx = get_max_plateau_gain(cum_gan, window=DEFAULT_PLATEAU_WINDOW)
                    score = float(gan_plateau)
                    subs_i = int(subs_idx)
                
                scores.append(score)
                subs.append(subs_i)
                preds.append(resultados["y_pred"])
                
                logger.info(f"Score {model_name} ({seed}): {score} con {subs_i} envíos")
        
        fi_df = pd.concat(feature_importances, axis=1)
        fi_mean = fi_df.mean(axis=1).sort_values(ascending=False)
        
        feature_importances_all[model_name] = fi_mean
        
        results[model_name] = {
            "scores": scores,
            "subs": subs,
            "preds": preds,
            "cum_gans": cum_gans,
            "y_true": resultados["y_true"],
        }
    
    # Simulación tipo leaderboard lunes style
    wins_summary = None
    if len(model_names) > 1:
        n_splits = 50
        
        wins_summary = {
            "public": {f"{model}_wins": 0 for model in model_names},
            "private": {f"{model}_wins": 0 for model in model_names},
        }
        
        first_model = model_names[0]
        y_true_full = np.array(results[first_model]["y_true"])
        
        for i, seed in enumerate(experiment_seeds):
            model_preds = {model: np.array(results[model]["preds"][i]) for model in model_names}
            
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
            for train_idx, test_idx in sss.split(np.zeros(len(y_true_full)), y_true_full):
                public_idx = test_idx
                private_idx = train_idx
                
                model_gains_public = {}
                model_gains_private = {}
                
                for model_name in model_names:
                    y_pred = model_preds[model_name]
                    cum_pub = get_cum_gan(y_pred[public_idx], y_true_full[public_idx])
                    cum_priv = get_cum_gan(y_pred[private_idx], y_true_full[private_idx])
                    gan_pub, _ = get_max_plateau_gain(cum_pub, window=PUBLIC_PLATEAU_WINDOW)
                    gan_priv, _ = get_max_plateau_gain(cum_priv, window=PRIVATE_PLATEAU_WINDOW)
                    model_gains_public[model_name] = float(gan_pub)
                    model_gains_private[model_name] = float(gan_priv)
                
                best_public = max(model_gains_public.items(), key=lambda x: x[1])
                best_private = max(model_gains_private.items(), key=lambda x: x[1])
                
                second_best_public = max([g for m, g in model_gains_public.items() if m != best_public[0]], default=float('-inf'))
                if best_public[1] > second_best_public:
                    wins_summary["public"][f"{best_public[0]}_wins"] += 1
                
                second_best_private = max([g for m, g in model_gains_private.items() if m != best_private[0]], default=float('-inf'))
                if best_private[1] > second_best_private:
                    wins_summary["private"][f"{best_private[0]}_wins"] += 1
    else:
        logger.info("Solo hay un modelo, omitiendo simulación de leaderboard (public/private)")
    
    # Gráfico de distribución de scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_scores = []
    for model_name, data in results.items():
        all_scores.extend(data["scores"])
    
    if len(all_scores) > 0:
        min_score = min(all_scores)
        max_score = max(all_scores)
        bins = np.linspace(min_score, max_score, 20)
    else:
        bins = 15
    
    model_stats = []
    for idx, (model_name, data) in enumerate(results.items()):
        scores = np.array(data["scores"])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        model_stats.append((model_name, mean_score, std_score))
        
        ax.hist(scores, bins=bins, alpha=0.5, label=f"{model_name}", edgecolor='black')
        ax.axvline(mean_score, linestyle='--', linewidth=2, 
                   label=f"{model_name} mean: {mean_score:.4f}")
    
    y_max = ax.get_ylim()[1]
    for idx, (model_name, mean_score, std_score) in enumerate(model_stats):
        y_pos = y_max * (0.95 - idx * 0.1)
        ax.text(mean_score, y_pos, 
                f"μ={mean_score:.4f}\nσ={std_score:.4f}", 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(f"Distribución de scores - {EXPERIMENT_NAME}", fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(LOGS_DIR, f"scores_dist_{EXPERIMENT_NAME}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de distribución guardado en {plot_path}")
    
    # Gráficos de ganancia acumulada (uno por modelo)
    for model_name, data in results.items():
        cum_gans = data["cum_gans"]
        
        if not cum_gans:
            logger.warning(f"No hay ganancias acumuladas para {model_name}, omitiendo gráfico")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        max_len = max(len(cg) for cg in cum_gans)
        max_envios = min(30000, max_len)
        
        cum_gans_aligned = []
        for cum_gan in cum_gans:
            if len(cum_gan) < max_envios:
                padded = np.pad(cum_gan, (0, max_envios - len(cum_gan)), mode='edge')
                cum_gans_aligned.append(padded)
            else:
                cum_gans_aligned.append(cum_gan[:max_envios])
        
        cum_gan_promedio = np.mean(cum_gans_aligned, axis=0)
        envios_promedio = np.arange(1, len(cum_gan_promedio) + 1)
        
        for i, (seed, cum_gan) in enumerate(zip(experiment_seeds, cum_gans)):
            envios = np.arange(1, len(cum_gan) + 1)
            ax.plot(envios[:max_envios], cum_gan[:max_envios], alpha=0.7)
        
        ax.plot(envios_promedio, cum_gan_promedio, color='black', linewidth=2, label='Promedio')
        
        ax.set_xlabel('Envíos', fontsize=12)
        ax.set_ylabel('Ganancia Acumulada', fontsize=12)
        ax.set_title(f'Ganancia Acumulada - {model_name} ({EXPERIMENT_NAME})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(LOGS_DIR, f"cumgan_{model_name}_{EXPERIMENT_NAME}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de ganancia acumulada guardado en {plot_path}")
        
        # Guardar CSV con curvas de ganancia
        cumgan_df = pd.DataFrame({f"seed_{seed}": cum_gan for seed, cum_gan in zip(experiment_seeds, cum_gans)})
        cumgan_df.index = np.arange(1, len(cumgan_df) + 1)
        cumgan_df.index.name = "envios"
        
        csv_filename = f"cumgan_data_{model_name}_{timestamp}.csv"
        csv_path = os.path.join(LOGS_DIR, csv_filename)
        cumgan_df.to_csv(csv_path)
        logger.info(f"Datos de ganancia acumulada guardados en {csv_path}")
        
        # Calcular y loggear número óptimo de submissions
        optimos = {}
        ganancias_maximas = {}
        
        for seed, cum_gan in zip(experiment_seeds, cum_gans):
            gan_max, idx_opt = get_max_plateau_gain(cum_gan, window=DEFAULT_PLATEAU_WINDOW)
            optimos[seed] = int(idx_opt) + 1
            ganancias_maximas[seed] = float(gan_max)
        
        promedio_opt = float(np.mean(list(optimos.values())))
        std_opt = float(np.std(list(optimos.values())))
        promedio_gan_max = float(np.mean(list(ganancias_maximas.values())))
        std_gan_max = float(np.std(list(ganancias_maximas.values())))
        
        detalles = " | ".join([f"Seed {seed}: {valor} envíos" for seed, valor in optimos.items()])
        logger.info(f"{model_name} - Envíos óptimos: {detalles} || Promedio: {promedio_opt:.1f} || Std: {std_opt:.1f}")
        
        detalles_gan = " | ".join([f"Seed {seed}: {valor:.4f}" for seed, valor in ganancias_maximas.items()])
        logger.info(f"{model_name} - Ganancia máxima: {detalles_gan} || Promedio: {promedio_gan_max:.4f} || Std: {std_gan_max:.4f}")
        
        results[model_name]["optimos"] = optimos
        results[model_name]["ganancias_maximas"] = ganancias_maximas
        results[model_name]["promedio_opt"] = promedio_opt
        results[model_name]["std_opt"] = std_opt
        results[model_name]["promedio_gan_max"] = promedio_gan_max
        results[model_name]["std_gan_max"] = std_gan_max
    
    # Resumen YAML
    timestamp = datetime.datetime.now(tz_ba).strftime("%Y-%m-%d_%H-%M-%S")
    yaml_path = os.path.join(LOGS_DIR, f"{EXPERIMENT_NAME}_{timestamp}.yaml")
    summary = {
        "timestamp": timestamp,
        "experiment_name": EXPERIMENT_NAME,
        "n_experiments": N_EXPERIMENTS,
        "val_month": VAL_MONTH,
        "seeds": experiment_seeds,
        "models": {},
    }
    
    if len(model_names) > 1:
        summary["splits_per_seed"] = n_splits
        summary["leaderboard_simulation"] = wins_summary
    
    for model_name, data in results.items():
        summary["models"][model_name] = {
            "avg_score": float(np.mean(data["scores"])),
            "std_score": float(np.std(data["scores"])),
            "avg_subs": float(np.mean(data["subs"])),
            "std_subs": float(np.std(data["subs"])),
            "scores": np.array(data["scores"], dtype=float).tolist(),
            "subs": np.array(data["subs"], dtype=int).tolist(),
            "params": MODELS_CONFIG[model_name]["params"],
            "features": MODELS_CONFIG[model_name]["chosen_features"],
            "months": MODELS_CONFIG[model_name]["months"],
            "semillerio": semillerio_por_modelo.get(model_name, 1),
            "avg_max_gain": data.get("promedio_gan_max", None),
            "std_max_gain": data.get("std_gan_max", None),
            "avg_optimal_subs": data.get("promedio_opt", None),
            "std_optimal_subs": data.get("std_opt", None),
        }
        
        feature_importances_all[model_name].to_csv(os.path.join(LOGS_DIR, f"fi_{model_name}_{timestamp}.csv"))
    
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, sort_keys=False, allow_unicode=True)
    
    logger.info(f"{YELLOW}Resumen guardado en {yaml_path}{RESET}")

