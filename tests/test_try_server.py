import json
import socket
import threading
import time

from cli.text2bash import try_server


def test_try_server_returns_none_when_socket_missing(tmp_path) -> None:
    socket_path = tmp_path / "does-not-exist.sock"
    assert try_server("list files", socket_path, 48) is None


def test_try_server_parses_fake_server_reply(tmp_path) -> None:
    socket_path = tmp_path / "server.sock"
    reply = {"command": "ls -la", "valid_bash": True}

    def serve_one() -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(socket_path))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall((json.dumps(reply) + "\n").encode())
        conn.close()
        srv.close()

    thread = threading.Thread(target=serve_one, daemon=True)
    thread.start()
    for _ in range(200):
        if socket_path.exists():
            break
        time.sleep(0.01)

    result = try_server("list files", socket_path, 48)
    thread.join(timeout=5)

    assert result == ("ls -la", True)
