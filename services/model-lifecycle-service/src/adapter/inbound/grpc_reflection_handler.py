from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import grpc
from google.protobuf import descriptor as _descriptor

from src.proto.grpc.reflection.v1alpha import reflection_pb2
from src.proto.grpc.reflection.v1alpha import reflection_pb2_grpc


class ServerReflectionService(reflection_pb2_grpc.ServerReflectionServicer):
    def __init__(
        self,
        *,
        service_names: Sequence[str],
        descriptor_sources: Sequence[_descriptor.FileDescriptor],
    ) -> None:
        self._service_names = list(service_names)
        self._descriptor_sources = list(descriptor_sources)

    def ServerReflectionInfo(
        self,
        request_iterator: Iterable[reflection_pb2.ServerReflectionRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[reflection_pb2.ServerReflectionResponse]:
        for request in request_iterator:
            yield self._handle_request(request, context)

    def _handle_request(
        self,
        request: reflection_pb2.ServerReflectionRequest,
        context: grpc.ServicerContext,
    ) -> reflection_pb2.ServerReflectionResponse:
        response = reflection_pb2.ServerReflectionResponse(
            valid_host=request.host,
            original_request=request,
        )

        request_type = request.WhichOneof("message_request")
        if request_type == "list_services":
            response.list_services_response.CopyFrom(
                reflection_pb2.ListServiceResponse(
                    service=[
                        reflection_pb2.ServiceResponse(name=name)
                        for name in self._service_names
                    ]
                )
            )
            return response

        if request_type == "file_by_filename":
            return self._fill_file_descriptor_response(
                response=response,
                context=context,
                resolver=lambda pool: pool.FindFileByName(request.file_by_filename),
            )

        if request_type == "file_containing_symbol":
            return self._fill_file_descriptor_response(
                response=response,
                context=context,
                resolver=lambda pool: pool.FindFileContainingSymbol(
                    self._normalize_symbol(request.file_containing_symbol)
                ),
            )

        if request_type == "all_extension_numbers_of_type":
            response.all_extension_numbers_response.CopyFrom(
                reflection_pb2.ExtensionNumberResponse(
                    base_type_name=request.all_extension_numbers_of_type,
                    extension_number=[],
                )
            )
            return response

        if request_type == "file_containing_extension":
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Reflection extension lookup is not implemented")
            response.error_response.CopyFrom(
                reflection_pb2.ErrorResponse(
                    error_code=grpc.StatusCode.UNIMPLEMENTED.value[0],
                    error_message="Reflection extension lookup is not implemented",
                )
            )
            return response

        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("Unknown reflection request")
        response.error_response.CopyFrom(
            reflection_pb2.ErrorResponse(
                error_code=grpc.StatusCode.INVALID_ARGUMENT.value[0],
                error_message="Unknown reflection request",
            )
        )
        return response

    def _fill_file_descriptor_response(
        self,
        *,
        response: reflection_pb2.ServerReflectionResponse,
        context: grpc.ServicerContext,
        resolver,
    ) -> reflection_pb2.ServerReflectionResponse:
        for pool in self._descriptor_pools():
            try:
                file_descriptor = resolver(pool)
            except KeyError:
                continue

            response.file_descriptor_response.CopyFrom(
                reflection_pb2.FileDescriptorResponse(
                    file_descriptor_proto=self._collect_file_descriptor_bytes(file_descriptor)
                )
            )
            return response

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Requested descriptor was not found")
        response.error_response.CopyFrom(
            reflection_pb2.ErrorResponse(
                error_code=grpc.StatusCode.NOT_FOUND.value[0],
                error_message="Requested descriptor was not found",
            )
        )
        return response

    def _descriptor_pools(self) -> Sequence[_descriptor.DescriptorPool]:
        return [source.pool for source in self._descriptor_sources]

    def _collect_file_descriptor_bytes(
        self, file_descriptor: _descriptor.FileDescriptor
    ) -> list[bytes]:
        collected: list[bytes] = []
        seen: set[str] = set()

        def visit(node: _descriptor.FileDescriptor) -> None:
            if node.name in seen:
                return
            seen.add(node.name)
            collected.append(node.serialized_pb)
            for dependency in node.dependencies:
                visit(dependency)

        visit(file_descriptor)
        return collected

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol[1:] if symbol.startswith(".") else symbol


def register_server_reflection(
    server: grpc.Server,
    *,
    service_names: Sequence[str],
    descriptor_sources: Sequence[_descriptor.FileDescriptor],
) -> None:
    reflection_pb2_grpc.add_ServerReflectionServicer_to_server(
        ServerReflectionService(
            service_names=service_names,
            descriptor_sources=descriptor_sources,
        ),
        server,
    )
