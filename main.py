import argparse
import dataclasses
import gzip
import json
import logging
import os
import random
import string
from typing import Any, Dict, Generator, List, Optional, TextIO, Union

import yaml

try:
    from yaml import CLoader as YamlLoader, CDumper as YamlDumper
except ImportError:
    from yaml import Loader as YamlLoader, Dumper as YamlDumper


logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

JSONL_CONTENT_TYPES = ('application/jsonl',)
JSON_CONTENT_TYPES = ('application/json', 'application/ld+json', 'text/json')
XML_CONTENT_TYPES = ('application/xml', 'text/xml')
CSV_CONTENT_TYPES = ('application/csv', 'text/csv')
GZ_CONTENT_TYPES = ('application/gzip',)
ZIP_CONTENT_TYPES = ('application/zip',)

BYTES_HEADER_PREFIX = '{<[bytes]>}'


class Headers:
    def __init__(
        self,
        value: Dict[str, Union[List[str], List[bytes]]],
    ) -> None:
        self.value = value
        self._ci_keys_map = {key.lower(): key for key in value.keys()}

    def has_content_type(self, content_type: str) -> bool:
        header_key = self._ci_keys_map.get('content-type')
        if not header_key:
            return False

        header_values = self.value[header_key]
        if not header_values:
            return False

        if isinstance(header_values[0], bytes):
            content_type = content_type.encode()

        for header_value in header_values:
            if isinstance(header_value, bytes):
                header_value = header_value.decode()

            return any(
                content_type == part
                for part in header_value.split(';')
            )

    def has_content_encoding(self, content_encoding: str) -> bool:
        header_key = self._ci_keys_map.get('content-encoding')
        if not header_key:
            return False

        header_values = self.value[header_key]
        if not header_values:
            return False

        for header_value in header_values:
            if isinstance(header_value, bytes):
                header_value = header_value.decode()

            if content_encoding == header_value:
                return True


@dataclasses.dataclass
class Request:
    method: str
    uri: str
    body: Optional[Union[str, bytes]]
    headers: Headers


@dataclasses.dataclass
class ResponseStatus:
    code: int
    message: str


@dataclasses.dataclass
class StringBody:
    value: Union[str, bytes]


@dataclasses.dataclass
class Response:
    headers: Headers
    body: Optional[StringBody]
    status: ResponseStatus


@dataclasses.dataclass
class Interaction:
    request: Request
    response: Response


@dataclasses.dataclass
class Cassette:
    interactions: List[Interaction]
    version: int


def tidy_json_body(body: str) -> str:
    if not body:
        return body

    body_json = json.loads(body)

    return json.dumps(body_json)


class TidyJsonBodiesInCassette:
    def __call__(self, cassette: Cassette) -> None:
        for interaction in cassette.interactions:
            self._for_request(interaction.request)
            self._for_response(interaction.response)

    def _for_request(self, request: Request) -> None:
        if any(
            request.headers.has_content_type(content_type)
            for content_type in JSON_CONTENT_TYPES
        ):
            request.body = tidy_json_body(request.body)
            return

    def _for_response(self, response: Response) -> None:
        if any(
            response.headers.has_content_type(content_type)
            for content_type in JSON_CONTENT_TYPES
        ):
            response.body.value = tidy_json_body(response.body.value)
            return


tidy_json_bodies_in_cassette = TidyJsonBodiesInCassette()


class LoadPackedCassette:
    def __call__(self, path: str) -> Cassette:
        self._check_extension(path)

        with self._open(path) as file_obj:
            cassette_data = yaml.load(file_obj, yaml.Loader)

            return self._load_cassette(cassette_data)

    def _check_extension(self, path: str) -> None:
        if not any(
            path.endswith(ending)
            for ending in ('.yaml.gz', '.yaml', '.yml', '.yml.gz')
        ):
            raise Exception(f'File is not cassette: {path}')

    def _open(self, path: str) -> TextIO:
        if path.endswith('.gz'):
            return gzip.open(path)

        return open(path)

    def _load_cassette(self, cassette: Dict[str, Any]) -> 'Cassette':
        return Cassette(
            interactions=[
                self._load_interaction(interaction)
                for interaction in cassette['interactions']
            ],
            version=cassette.get('version', 1),
        )

    def _load_interaction(
        self,
        interaction: Dict[str, Any],
    ) -> 'Interaction':
        return Interaction(
            request=self._load_request(interaction['request']),
            response=self._load_response(interaction['response']),
        )

    def _load_request(self, request: Dict[str, Any]) -> 'Request':
        return Request(
            method=request['method'],
            uri=request['uri'],
            body=request['body'],
            headers=self._load_headers(request['headers']),
        )

    def _load_response(self, response: Dict[str, Any]) -> 'Response':
        return Response(
            body=self._load_response_body(response['body']),
            headers=self._load_headers(response['headers']),
            status=self._load_response_status(response['status']),
        )

    def _load_response_body(self, body: Dict[str, Any]):
        if 'string' not in body:
            raise Exception(f'Unexpected body: {list(body)}')

        return StringBody(body['string'])

    def _load_headers(
        self,
        headers: Dict[str, Union[List[str], List[bytes]]],
    ) -> 'Headers':
        return Headers(headers)

    def _load_response_status(self, status: Dict[str, Any]) -> 'ResponseStatus':
        return ResponseStatus(
            code=status['code'],
            message=status['message'],
        )


load_packed_cassette = LoadPackedCassette()


def dump_to_yaml_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, 'w') as file_obj:
        yaml.dump(data, file_obj, Dumper=YamlDumper)
        file_obj.flush()


def load_yaml_from_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as file_obj:
            return yaml.load(file_obj, Loader=YamlLoader)
    except yaml.YAMLError as e:
        raise Exception(f'Error in {path}: {e}')


def dump_to_file(path: str, data: Union[str, bytes]) -> None:
    if isinstance(data, str):
        file_obj = open(path, mode='wt', newline='', encoding='utf-8')
    else:
        file_obj = open(path, mode='wb')

    with file_obj:
        file_obj.write(data)
        file_obj.flush()


def load_str_from_file(path: str) -> str:
    with open(path, mode='rt', newline='', encoding='utf-8') as file_obj:
        return file_obj.read()


def load_bytes_from_file(path: str) -> bytes:
    with open(path, 'rb') as file_obj:
        return file_obj.read()


def generate_random_string(length: int) -> str:
    alphabet = string.ascii_letters + string.digits

    return ''.join(random.choice(alphabet) for _ in range(length))


class DumpCassetteToUnpackedForm:
    def __call__(
        self,
        root_dir: str,
        cassette: Cassette,
    ) -> None:
        self._prepare_root(root_dir)

        for number, interaction in enumerate(cassette.interactions):
            interaction_dir = self._prepare_interaction_dir(
                base_dir=root_dir,
                interaction=interaction,
                number=number,
            )
            self._dump_request(
                base_dir=interaction_dir,
                request=interaction.request,
            )
            self._dump_response(
                base_dir=interaction_dir,
                response=interaction.response,
            )

        dump_to_file(
            path=os.path.join(root_dir, 'version.txt'),
            data=str(cassette.version),
        )

    def _prepare_root(self, path: str) -> None:
        if (
            os.path.exists(path)
            and len(list(os.listdir(path))) > 0
        ):
            raise Exception(f'Directory not empty: {path}')

        os.makedirs(path, exist_ok=True)

    def _prepare_interaction_dir(
        self,
        base_dir: str,
        interaction: Interaction,
        number: int
    ) -> str:
        http_method = interaction.request.method
        status_code = interaction.response.status.code

        interaction_dir = os.path.join(
            base_dir,
            f'{number:06d}_{http_method}_{status_code}',
        )
        os.mkdir(interaction_dir)

        return interaction_dir

    def _dump_request(
        self,
        base_dir: str,
        request: Request,
    ) -> None:
        dump_to_file(
            path=os.path.join(base_dir, f'__uri.txt'),
            data=request.uri,
        )
        self._dump_headers(
            request.headers,
            path=os.path.join(base_dir, f'_headers.yaml'),
        )
        dump_to_file(
            path=os.path.join(base_dir, f'_method.txt'),
            data=request.method,
        )

        if request.body:
            self._dump_body(
                base_dir=base_dir,
                filename='_body',
                body=request.body,
                headers=request.headers,
            )

    def _dump_response(
        self,
        base_dir: str,
        response: Response,
    ) -> None:
        self._dump_headers(
            response.headers,
            path=os.path.join(base_dir, 'headers.yaml'),
        )
        self._dump_response_status(
            response.status,
            path=os.path.join(base_dir, 'status.yaml'),
        )
        self._dump_body(
            base_dir=base_dir,
            filename='body',
            body=response.body.value,
            headers=response.headers,
        )

    def _dump_headers(
        self,
        headers: Headers,
        path: str,
    ) -> None:
        headers_data = {
            header_name: [
                self._try_to_convert_header_value_to_str(header_name, value)
                for value in values
            ]
            for header_name, values in headers.value.items()
        }
        dump_to_yaml_file(path, data=headers_data)

    def _try_to_convert_header_value_to_str(
        self,
        header_name: str,
        value: Union[bytes, str],
    ) -> Union[bytes, str]:
        if not isinstance(value, bytes):
            return value

        try:
            return BYTES_HEADER_PREFIX + value.decode('utf-8')
        except ValueError:
            logger.debug("Can't decode to utf-8: %s=%r", header_name, value)

        return value

    def _dump_response_status(
        self,
        status: ResponseStatus,
        path: str,
    ) -> None:
        data = {
            'code': status.code,
            'message': status.message,
        }
        dump_to_yaml_file(path, data)

    def _dump_body(
        self,
        base_dir: str,
        filename: str,
        body: Union[str, bytes],
        headers: Headers,
    ) -> None:
        ext = self._get_filename_ext(body, headers)

        if (
            self._is_text_format(ext)
            and isinstance(body, bytes)
            and headers.has_content_encoding('gzip')
        ):
            ext = f'{ext}.gz'

        if ext == 'json':
            body = self._indent_json(body)

        dump_to_file(
            path=os.path.join(base_dir, f'{filename}.{ext}'),
            data=body,
        )

    def _get_filename_ext(
        self,
        body: Union[str, bytes],
        headers: Headers,
    ) -> str:
        if any(
            headers.has_content_type(content_type)
            for content_type in JSONL_CONTENT_TYPES
        ):
            return 'jsonl'

        if any(
            headers.has_content_type(content_type)
            for content_type in JSON_CONTENT_TYPES
        ):
            return 'json'

        if any(
            headers.has_content_type(content_type)
            for content_type in XML_CONTENT_TYPES
        ):
            return 'xml'

        if any(
            headers.has_content_type(content_type)
            for content_type in CSV_CONTENT_TYPES
        ):
            return 'csv'

        if any(
            headers.has_content_type(content_type)
            for content_type in GZ_CONTENT_TYPES
        ):
            return 'gz'

        if any(
            headers.has_content_type(content_type)
            for content_type in ZIP_CONTENT_TYPES
        ):
            return 'zip'

        if isinstance(body, bytes):
            return 'bin'

        return 'txt'

    def _is_text_format(self, ext: str) -> bool:
        return ext in ('txt', 'json', 'csv', 'xml')

    def _indent_json(self, body: str) -> str:
        if not body:
            return body

        json_body = json.loads(body)

        return json.dumps(
            json_body,
            indent=4,
            separators=(',', ': '),
            cls=json.JSONEncoder,
        )


dump_cassette_to_unpacked_form = DumpCassetteToUnpackedForm()


class LoadUnpackedCassette:
    def __call__(self, root_dir: str) -> Cassette:
        interactions = list(self._load_interactions(root_dir))
        version = int(load_str_from_file(
            path=os.path.join(root_dir, 'version.txt'))
        )

        return Cassette(
            interactions=interactions,
            version=version,
        )

    def _load_interactions(
        self,
        root_dir: str,
    ) -> Generator[Interaction, None, None]:
        for interaction_dir in self._iterate_over_interactions(root_dir):
            request = self._load_request(base_dir=interaction_dir)
            response = self._load_response(base_dir=interaction_dir)

            yield Interaction(request, response)

    def _iterate_over_interactions(
        self,
        root_dir: str,
    ) -> Generator[str, None, None]:
        names = sorted(os.listdir(root_dir))

        for name in names:
            if name == 'version.txt':
                continue
            path = os.path.join(root_dir, name)

            if not os.path.isdir(path):
                logger.info('Not an interaction directory: %s', path)
                continue

            yield path

    def _load_request(self, base_dir: str) -> Request:
        uri = load_str_from_file(
            path=os.path.join(base_dir, '__uri.txt')
        )
        method = self._load_method(
            path=os.path.join(base_dir, '_method.txt')
        )
        headers = self._load_headers(
            path=os.path.join(base_dir, '_headers.yaml'),
        )

        body_file_name = self._get_body_filename(
            base_dir,
            filename='_body',
        )
        body = self._load_body(
            base_dir=base_dir,
            filename=body_file_name,
        )

        return Request(
            method=method,
            uri=uri,
            headers=headers,
            body=body,
        )

    def _load_response(self, base_dir: str) -> Response:
        headers = self._load_headers(
            path=os.path.join(base_dir, 'headers.yaml')
        )
        body_file_name = self._get_body_filename(
            base_dir,
            filename='body',
        )
        body = self._load_body(
            base_dir=base_dir,
            filename=body_file_name,
        )
        if body is not None:
            body = StringBody(body)

        status = self._load_response_status(
            path=os.path.join(base_dir, 'status.yaml')
        )

        return Response(
            headers=headers,
            body=body,
            status=status,
        )

    def _load_headers(self, path: str) -> Headers:
        headers_data = load_yaml_from_file(path)
        _headers_data = {
            header_name: [
                self._convert_header_value_to_original(value)
                for value in values
            ]
            for header_name, values in headers_data.items()
        }

        return Headers(_headers_data)

    def _convert_header_value_to_original(
        self,
        value: Union[bytes, str],
    ) -> Union[bytes, str]:
        if isinstance(value, bytes):
            return value

        if value.startswith(BYTES_HEADER_PREFIX):
            return value[len(BYTES_HEADER_PREFIX):].encode('utf-8')

        return value

    def _load_method(self, path: str) -> str:
        method = load_str_from_file(path)
        method_ = method.strip().upper()

        if method_ not in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE'):
            raise Exception(f'Unexpected http method: {path}')

        return method

    def _get_body_filename(
        self,
        base_dir: str,
        filename: str,
    ) -> Optional[str]:
        names = os.listdir(base_dir)
        prefix = f'{filename}.'

        body_names = [
            name for name in names
            if name.startswith(prefix)
        ]

        if len(body_names) > 1:
            raise Exception(f'Too many `{prefix}...` files in {base_dir}')

        if len(body_names) == 0:
            return None

        return body_names[0]

    def _load_body(
        self,
        base_dir: str,
        filename: Optional[str],
    ) -> Optional[Union[str, bytes]]:
        if filename is None:
            return None

        return self._load_body_data(
            path=os.path.join(base_dir, filename)
        )

    def _load_body_data(
        self,
        path: str,
    ) -> Union[str, bytes]:
        if any(
            path.endswith(ext)
            for ext in ('.bin', '.zip', '.xlsx', '.xls', '.gz')
        ):
            return load_bytes_from_file(path)

        if path.endswith('.json'):
            return tidy_json_body(load_str_from_file(path))

        if any(
            path.endswith(ext)
            for ext in ('.txt', '.csv', '.csv', '.xml')
        ):
            return load_str_from_file(path)

        raise Exception(f'Unexpected body: {path}')

    def _load_response_status(
        self,
        path: str,
    ) -> 'ResponseStatus':
        status = load_yaml_from_file(path)

        return ResponseStatus(
            code=status['code'],
            message=status['message'],
        )


load_unpacked_cassette = LoadUnpackedCassette()


class DumpCassetteToPackedForm:
    def __call__(
        self,
        cassette: Cassette,
        path: str,
    ) -> None:
        cassette_data = self._dump_cassette(cassette)

        with self._open(path) as file_obj:
            yaml.dump(cassette_data, file_obj, Dumper=YamlDumper)
            file_obj.flush()

    def _open(self, path: str) -> TextIO:
        if path.endswith('.gz'):
            return gzip.open(path, 'wt', compresslevel=9)

        return open(path, 'wt')

    def _dump_cassette(self, cassette: Cassette) -> Dict[str, Any]:
        return {
            'interactions': [
                self._dump_interaction(interaction)
                for interaction in cassette.interactions
            ],
            'version': cassette.version,
        }

    def _dump_interaction(
        self,
        interaction: Interaction,
    ) -> Dict[str, Any]:
        return {
            'request': self._dump_request(interaction.request),
            'response': self._dump_response(interaction.response),
        }

    def _dump_request(self, request: Request) -> Dict[str, Any]:
        return {
            'method': request.method,
            'uri': request.uri,
            'body': request.body,
            'headers': request.headers.value,
        }

    def _dump_response(self, response: Response) -> Dict[str, Any]:
        return {
            'body': self._dump_response_body(response.body),
            'headers': response.headers.value,
            'status': self._dump_response_status(response.status),
        }

    def _dump_response_body(
        self,
        body: StringBody,
    ) -> Dict[str, Any]:
        return {'string': body.value}

    def _dump_response_status(
        self,
        status: ResponseStatus,
    ) -> Dict[str, Any]:
        return {
            'code': status.code,
            'message': status.message,
        }


dump_cassette_to_packed_form = DumpCassetteToPackedForm()


def generate_unpacked_cassette_path(cassette_path: str) -> str:
    cassette_dir = os.path.dirname(cassette_path)
    base_name = os.path.basename(cassette_path)

    base_name = _clean_from_extensions(base_name)
    random_part = generate_random_string(length=6)

    return os.path.join(cassette_dir, f'{base_name}__{random_part}')


def generate_cassette_path(unpacked_cassette_path: str) -> str:
    cassette_dir = os.path.dirname(unpacked_cassette_path)
    base_name = os.path.basename(unpacked_cassette_path)

    base_name = _clean_from_extensions(base_name)
    random_part = generate_random_string(length=6)

    return os.path.join(cassette_dir, f'{base_name}_{random_part}.yaml.gz')


def _clean_from_extensions(base_name: str) -> str:
    for s in ('.gz', '.yaml', '.yml'):
        base_name = base_name.replace(s, '')

    return base_name


def perform_unpack(
    source_path: str,
    dest_path: Optional[str] = None,
) -> None:
    if not dest_path:
        dest_path = generate_unpacked_cassette_path(source_path)

    cassette = load_packed_cassette(source_path)
    dump_cassette_to_unpacked_form(dest_path, cassette)


def perform_pack(
    source_path: str,
    dest_path: Optional[str] = None,
) -> None:
    if not dest_path:
        dest_path = generate_cassette_path(source_path)

    cassette = load_unpacked_cassette(source_path)
    dump_cassette_to_packed_form(cassette, dest_path)


def perform_tidy_json_bodies(
    source_path: str,
    dest_path: Optional[str] = None,
) -> None:
    if not dest_path:
        dest_path = generate_cassette_path(source_path)

    cassette = load_packed_cassette(source_path)
    tidy_json_bodies_in_cassette(cassette)
    dump_cassette_to_packed_form(cassette, dest_path)


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description='VCR cassette converter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        help='Command to perform',
        dest='command',
        required=True,
    )

    unpack_parser = subparsers.add_parser(
        'unpack',
        help='Cassette splitting',
    )
    unpack_parser.add_argument(
        '--path',
        dest='source_path',
        type=str,
        help='Path to source',
        required=True,
    )
    unpack_parser.add_argument(
        '--dest-path',
        dest='dest_path',
        type=str,
        help='Path to destination',
        required=False,
    )

    pack_parser = subparsers.add_parser(
        'pack',
        help='Pack cassette',
    )
    pack_parser.add_argument(
        '--path',
        dest='source_path',
        type=str,
        help='Path to source',
        required=True,
    )
    pack_parser.add_argument(
        '--dest-path',
        dest='dest_path',
        type=str,
        help='Path to destination',
        required=False,
    )

    tidy_json_parser = subparsers.add_parser(
        'tidy-json-bodies',
        help='Tidy JSON bodies in cassette',
    )
    tidy_json_parser.add_argument(
        '--path',
        dest='source_path',
        type=str,
        help='Path to source',
        required=True,
    )
    tidy_json_parser.add_argument(
        '--dest-path',
        dest='dest_path',
        type=str,
        help='Path to destination',
        required=False,
    )

    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()

    if args['command'] == 'unpack':
        perform_unpack(
            source_path=args['source_path'],
            dest_path=args.get('dest_path'),
        )
        return

    if args['command'] == 'pack':
        perform_pack(
            source_path=args['source_path'],
            dest_path=args.get('dest_path'),
        )
        return

    if args['command'] == 'tidy-json-bodies':
        perform_tidy_json_bodies(
            source_path=args['source_path'],
            dest_path=args.get('dest_path'),
        )
        return

    raise Exception(f'Unexpected command: {args}')


if __name__ == '__main__':
    main()
