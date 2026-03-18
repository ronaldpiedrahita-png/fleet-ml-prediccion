#!/bin/bash
echo "Esperando base de datos..."
sleep 5
echo "Inicializando datos..."
python 01_fleet_db_setup.py
echo "Listo."