"""
Bitmex REST API Library

Timestamp 입력시 Format: "2014-12-26 11:00"

"""
import requests
import datetime
import time

class BitmexRestAPI(object):


    @staticmethod
    def get_candle_data(symbol, intv, start_dt, end_dt):
        """

        :param symbol: .BXBT, .BETH
        :param intv: 1m, 1h
        :param start_dt: '2010-01-01 11:00'
        :param end_dt:
        :return:
        """
        request_url = "https://www.bitmex.com/api/udf/history"

        start_dt = int(datetime.datetime.strptime(start_dt, '%Y-%m-%d %H:%M').strftime('%s'))
        end_dt = int(datetime.datetime.strptime(end_dt, '%Y-%m-%d %H:%M').strftime('%s'))
        intv = 1 if intv == '1m' else 60

        query_params = {
            'symbol': symbol,
            'resolution': intv,
            'from': start_dt,
            'to': end_dt
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params=query_params)
        return result


    @staticmethod
    def get_funding_rate_by_count(symbol, start_dt, cnt):
        request_url = 'https://www.bitmex.com/api/v1/funding'

        query_params = {
            'symbol': symbol,
            'startTime': start_dt,
            'count': cnt
        }
        headers = {
            'Accept': 'application/json'
        }

        result = BitmexRestAPI._request(request_url, 'GET', query_params, headers)

        return result

    @staticmethod
    def get_funding_rate(symbol, start_dt, end_dt):

        final_result = []

        while True:

            result = BitmexRestAPI.get_funding_rate_by_count(symbol, start_dt, 500)
            final_result += result

            last_dt = datetime.datetime.strptime(final_result[-1]['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z')

            if last_dt >= datetime.datetime.strptime(end_dt, '%Y-%m-%d %H:%M').astimezone(datetime.timezone.utc):

                return [i for i in final_result if datetime.datetime.strptime(i['timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z') <= datetime.datetime.strptime(end_dt, '%Y-%m-%d %H:%M').astimezone(datetime.timezone.utc)]
            else:

                start_dt = final_result[-1]


    @staticmethod
    def _request(request_url, action, query_params=None, headers=None):

        while True:

            response = getattr(requests, action.lower())(request_url, headers=headers, params=query_params, verify=False)

            if response.ok:
                return BitmexRestAPI._handle_response(response)
            if response.status_code == 429:
                time.sleep(10)



    @staticmethod
    def _handle_response(response):

        try:
            result = response.json()
            return result
        except ValueError:
            raise ValueError(response.text)


if __name__ == '__main__':

    c = BitmexRestAPI()
    c.get_funding_rate('XBTUSD', '2020-01-01 00:00', '2020-06-20 00:00')

    # c.get_funding

    print(1)