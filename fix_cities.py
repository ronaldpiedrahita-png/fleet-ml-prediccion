from sqlalchemy import create_engine, text

engine = create_engine("postgresql://postgres:Valeria2004@localhost/fleetdb")

with engine.connect() as conn:
    conn.execute(text("""
        UPDATE trucks
        SET base_city = 'Queretaro'
        WHERE base_city NOT IN ('Monterrey','CDMX','Guadalajara','Tijuana','Queretaro')
    """))
    conn.commit()
    print("Ciudades corregidas")