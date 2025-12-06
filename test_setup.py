import fastf1
fastf1.Cache.enable_cache('./cache')

session = fastf1.get_session(2024, 'Abu Dhabi', 'Q')
session.load()

print(session.results[['Abbreviation', 'Position']].head(5))