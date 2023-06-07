from pytwitter import Api
api = Api(client_id="LWVxSmFmXzlJWWRsWldBUExzWnM6MTpjaQ",client_secret='Zqu-7dgiwi-DxrPt6dy-gKGTZ_gEz_dHZ40yoh3nc_sGXSxEhO',oauth_flow=True)
url, code_verifier, resultado = api.get_oauth2_authorize_url()
print(f'url={url}\ncode_verifier={code_verifier}\nresultado={resultado}')
token = api.generate_oauth2_access_token('http://localhost/?state=state&code=code', code_verifier)
print(f'token = {token}')