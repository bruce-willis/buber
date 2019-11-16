import itertools
from typing import NamedTuple

import polyline
import requests
import requests_cache


class Coord(NamedTuple):
    lat: float
    lon: float


flatten = itertools.chain.from_iterable

requests_cache.install_cache('hsl_cache', backend='sqlite', expire_after=600, allowable_methods=('GET', 'POST'))
endpoint = 'https://api.digitransit.fi/routing/v1/routers/hsl/index/graphql'


def run_query(query):
    request = requests.post(endpoint, json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, request.text))


route_str = """
{{
  plan (
    from: {{ lat: {src.lat:.15f}, lon: {src.lon:.15f} }},
    to: {{ lat: {dest.lat:.15f}, lon: {dest.lon:.15f} }}
    {no_walk_modifier}
  ) {{
    from {{
      name
      lat
      lon
    }}
    to {{
      name
      lat
      lon
    }}
    itineraries {{
      startTime
      endTime
      duration
      waitingTime
      walkTime
      legs {{
        mode
        steps {{
          lat
          lon
        }}
        legGeometry {{
          points
        }}
        route {{
          shortName
          longName
          gtfsId
          url
        }}
      }}
    }}
  }}
}}
""".format


def route_query_fmt(src, dest, no_walk):
    no_walk_modifier = ", transportModes: [; {mode: BICYCLE}]" if no_walk else ""
    return route_str(no_walk_modifier=no_walk_modifier, src=src, dest=dest)


gpx_header = """
<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.topografix.com/GPX/1/1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd" version="1.1" creator="Buber">
"""
gpx_line = """<wpt lat="{lat}" lon="{lon}"></wpt>"""
gpx_footer = "</gpx>"


def route_gpx_fmt(coords):
    return gpx_header + '\n'.join(gpx_line.format(lat=c.lat, lon=c.lon) for c in coords) + gpx_footer


def query_route(src, dest, no_walk=True, full=False):
    res = run_query(route_query_fmt(src=src, dest=dest, no_walk=no_walk))
    try:
        routes = res['data']['plan']['itineraries']
    except KeyError:
        print('Inconsistent response')
        print(res)
        return []

    if routes:
        route = routes[0]
        if full:
            steps = [(Coord(*step)
                      for step in polyline.decode(leg['legGeometry']['points']))
                     for leg in route['legs']]
            return list(flatten(steps))
        else:
            transport_leg = next(filter(lambda leg: leg['mode'] != 'WALK', route['legs']), route['legs'][0])
            steps = [Coord(*step) for step in polyline.decode(transport_leg['legGeometry']['points'])]
            return steps
    else:
        return []


if __name__ == '__main__':
    result = query_route(Coord(60.184662, 24.825294), Coord(60.169563, 24.940038))
    print(result)
    print(route_gpx_fmt(result))
