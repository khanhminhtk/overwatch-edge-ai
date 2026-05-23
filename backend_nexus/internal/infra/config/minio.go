package config

type Minio struct {
	MinioServer MinioClient `yaml:"minio_server"`
}

type MinioClient struct {
	Endpoint          string             `yaml:"endpoint"`
	AccessKey         string             `yaml:"access_key"`
	SecretKey         string             `yaml:"secret_key"`
	SessionToken      string             `yaml:"session_token"`
	UseSSL            bool               `yaml:"use_ssl"`
	Region            string             `yaml:"region"`
	HealthCheck       bool               `yaml:"health_check"`
	InsecureSkipTLS   bool               `yaml:"insecure_skip_tls"`
	ConfigServiceGRPC MinioServiceGRPC   `yaml:"config_service_grpc"`
}

type MinioServiceGRPC struct {
	GRPCPort                string `yaml:"grpc_port"`
	JWTSecret               string `yaml:"jwt_secret"`
	RefreshToken            string `yaml:"refresh_token"`
	ExpiresTimeAccessToken  int64  `yaml:"expires_time_access_token"`
	ExpiresTimeURLUpload    int64  `yaml:"expires_time_url_upload"`
	ExpiresTimeURLDownload  int64  `yaml:"expires_time_url_download"`
}
